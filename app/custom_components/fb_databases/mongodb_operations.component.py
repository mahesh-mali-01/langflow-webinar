import logging
import time
from enum import Enum
from typing import Any, Dict, List

from langflow.custom import Component
from langflow.inputs import StrInput, SecretStrInput, DataInput, BoolInput, IntInput, DropdownInput
from langflow.io import Output
from langflow.schema import Data
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError


class CollectionName(str, Enum):
    """Available collection names for schema validation"""
    PROCESSED_VIDEOS = "processed_videos"


class MongoDBOperationsComponent(Component):
    display_name = "MongoDB Operations"
    description = "Perform MongoDB operations (insert, update, find, delete) with connection management"
    documentation: str = "Connect to MongoDB and perform database operations with comprehensive error handling"
    icon = "database"
    name = "MongoDBOperations"

    inputs = [
        SecretStrInput(
            name="connection_string",
            display_name="MongoDB Connection String",
            info="MongoDB connection string (e.g., mongodb://localhost:27017 or mongodb+srv://...)",
            required=True,
        ),
        StrInput(
            name="database_name",
            display_name="Database Name",
            info="Target database name",
            required=True,
        ),
        DropdownInput(
            name="collection_name",
            display_name="Collection Name",
            info="Target MongoDB collection name",
            options=[collection.value for collection in CollectionName],
            required=True,
        ),
        DropdownInput(
            name="operation",
            display_name="Operation",
            info="MongoDB operation to perform",
            options=["insert_one", "insert_many", "find_one", "find_many", "update_one", "update_many", "delete_one",
                     "delete_many"],
            value="insert_one",
            required=True,
        ),
        DataInput(
            name="document_data",
            display_name="Document Data",
            info="Document data for insert/update operations",
            required=False,
        ),
        DataInput(
            name="filter_query",
            display_name="Filter Query",
            info="MongoDB filter query for find/update/delete operations",
            required=False,
        ),
        DataInput(
            name="update_data",
            display_name="Update Data",
            info="Update data for update operations (use $set, $inc, etc.)",
            required=False,
        ),
        IntInput(
            name="connection_timeout",
            display_name="Connection Timeout (seconds)",
            info="Connection timeout in seconds",
            value=10,
            required=False,
        ),
        IntInput(
            name="server_selection_timeout",
            display_name="Server Selection Timeout (seconds)",
            info="Server selection timeout in seconds",
            value=100,
            required=False,
        ),
        IntInput(
            name="find_limit",
            display_name="Find Limit",
            info="Maximum number of documents to return for find operations",
            value=100,
            required=False,
        ),
        BoolInput(
            name="upsert",
            display_name="Upsert",
            info="Create document if it doesn't exist (for update operations)",
            value=False,
        ),
        BoolInput(
            name="bypass_document_validation",
            display_name="Bypass Document Validation",
            info="Skip document validation during insert/update",
            value=False,
            show=False,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Logging",
            info="Enable detailed logging",
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="Operation Result", name="operation_result", method="perform_operation"),
        Output(display_name="Success", name="success", method="get_success"),
        Output(display_name="Document Count", name="document_count", method="get_document_count"),
        Output(display_name="Error Details", name="error_details", method="get_error_details"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = None
        self.database = None
        self.collection = None
        self.operation_success = False
        self.operation_result = None
        self.error_details = None
        self.document_count = 0

        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _setup_logging(self):
        """Set up logging level based on verbose setting"""
        if hasattr(self, 'verbose'):
            self.logger.setLevel(logging.INFO if self.verbose else logging.ERROR)

    def _debug_log(self, message: str, level: str = "INFO"):
        """Custom logging function with configurable log level"""
        self._setup_logging()

        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        log_level = level_mapping.get(level.upper(), logging.INFO)
        self.logger.log(log_level, message)

        # Also log to Langflow's logger for visibility in UI
        if level in ["ERROR", "CRITICAL"] or (hasattr(self, 'verbose') and self.verbose):
            self.log(f"[{level}] {message}")

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import pymongo
            self._debug_log("PyMongo dependency check passed", "INFO")
        except ImportError:
            raise ImportError(
                "pymongo package is required. "
                "Please install it: pip install pymongo"
            )

    def _connect_to_mongodb(self):
        """Establish connection to MongoDB"""
        try:
            self._debug_log("Checking dependencies...", "INFO")
            self._check_dependencies()

            self._debug_log("Connecting to MongoDB...", "INFO")

            # Create client with timeout settings
            self.client = MongoClient(
                self.connection_string,
                connectTimeoutMS=self.connection_timeout * 1000,
                serverSelectionTimeoutMS=self.server_selection_timeout * 1000,
                retryWrites=True,
                w="majority"
            )

            # Test connection
            self.client.admin.command('ping')
            self._debug_log("Successfully connected to MongoDB", "INFO")

            # Get database and collection
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]

            self._debug_log(f"Connected to database: {self.database_name}, collection: {self.collection_name}", "INFO")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            error_msg = f"Failed to connect to MongoDB: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error connecting to MongoDB: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _extract_data(self, input_data: Any) -> Dict[str, Any]:
        """Extract data from various input formats"""
        self._debug_log(f"Extracting data from input...{input_data}", "INFO")
        if input_data is None:
            return {}

        try:
            # Handle Data object
            if hasattr(input_data, 'data'):
                data = input_data.data
                self._debug_log(f"Data object found, extracting data: {data}", "INFO")
                if isinstance(data, dict):
                    data = data.get('validated_data', {})
                return data
            else:
                data = input_data

            # Handle string JSON
            if isinstance(data, str):
                import json
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    raise ValueError("Input data must be valid JSON string or dictionary")

            # Handle list of dictionaries
            if isinstance(data, list):
                return data

            # Ensure we have a dictionary
            if not isinstance(data, dict):
                raise ValueError("Input data must be a dictionary or list of dictionaries")

            return data

        except Exception as e:
            self._debug_log(f"Error extracting data: {str(e)}", "ERROR")
            raise

    def _perform_insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform insert_one operation"""
        try:
            self._debug_log(f"Inserting single document with {len(document)} fields", "INFO")

            result = self.collection.insert_one(
                document,
                bypass_document_validation=self.bypass_document_validation
            )

            self.document_count = 1
            self._debug_log(f"Document inserted successfully with ID: {result.inserted_id}", "INFO")

            return {
                "operation": "insert_one",
                "success": True,
                "inserted_id": str(result.inserted_id),
                "acknowledged": result.acknowledged,
                "document_count": 1
            }

        except DuplicateKeyError as e:
            error_msg = f"Duplicate key error: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Insert operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_insert_many(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform insert_many operation"""
        try:
            self._debug_log(f"Inserting {len(documents)} documents", "INFO")

            result = self.collection.insert_many(
                documents,
                bypass_document_validation=self.bypass_document_validation
            )

            self.document_count = len(result.inserted_ids)
            self._debug_log(f"Successfully inserted {len(result.inserted_ids)} documents", "INFO")

            return {
                "operation": "insert_many",
                "success": True,
                "inserted_ids": [str(id) for id in result.inserted_ids],
                "acknowledged": result.acknowledged,
                "document_count": len(result.inserted_ids)
            }

        except DuplicateKeyError as e:
            error_msg = f"Duplicate key error: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Insert many operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_find_one(self, filter_query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform find_one operation"""
        try:
            self._debug_log(f"Finding single document with filter: {filter_query}", "INFO")

            result = self.collection.find_one(filter_query)

            if result:
                self.document_count = 1
                # Convert ObjectId to string for JSON serialization
                if '_id' in result:
                    result['_id'] = str(result['_id'])
                self._debug_log("Document found successfully", "INFO")
            else:
                self.document_count = 0
                self._debug_log("No document found matching the filter", "INFO")

            return {
                "operation": "find_one",
                "success": True,
                "document": result,
                "document_count": self.document_count
            }

        except Exception as e:
            error_msg = f"Find one operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_find_many(self, filter_query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform find_many operation"""
        try:
            self._debug_log(f"Finding documents with filter: {filter_query}, limit: {self.find_limit}", "INFO")

            cursor = self.collection.find(filter_query).limit(self.find_limit)
            documents = []

            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)

            self.document_count = len(documents)
            self._debug_log(f"Found {len(documents)} documents", "INFO")

            return {
                "operation": "find_many",
                "success": True,
                "documents": documents,
                "document_count": self.document_count,
                "limit_applied": self.find_limit
            }

        except Exception as e:
            error_msg = f"Find many operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_update_one(self, filter_query: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform update_one operation"""
        try:
            self._debug_log(f"Updating single document with filter: {filter_query}", "INFO")

            result = self.collection.update_one(
                filter_query,
                update_data,
                upsert=self.upsert,
                bypass_document_validation=self.bypass_document_validation
            )

            self.document_count = result.modified_count
            self._debug_log(
                f"Update operation completed - Modified: {result.modified_count}, Matched: {result.matched_count}",
                "INFO")

            response = {
                "operation": "update_one",
                "success": True,
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "acknowledged": result.acknowledged,
                "document_count": result.modified_count
            }

            if result.upserted_id:
                response["upserted_id"] = str(result.upserted_id)
                self._debug_log(f"Document upserted with ID: {result.upserted_id}", "INFO")

            return response

        except Exception as e:
            error_msg = f"Update one operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_update_many(self, filter_query: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform update_many operation"""
        try:
            self._debug_log(f"Updating multiple documents with filter: {filter_query}", "INFO")

            result = self.collection.update_many(
                filter_query,
                update_data,
                upsert=self.upsert,
                bypass_document_validation=self.bypass_document_validation
            )

            self.document_count = result.modified_count
            self._debug_log(
                f"Update operation completed - Modified: {result.modified_count}, Matched: {result.matched_count}",
                "INFO")

            response = {
                "operation": "update_many",
                "success": True,
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
                "acknowledged": result.acknowledged,
                "document_count": result.modified_count
            }

            if result.upserted_id:
                response["upserted_id"] = str(result.upserted_id)
                self._debug_log(f"Document upserted with ID: {result.upserted_id}", "INFO")

            return response

        except Exception as e:
            error_msg = f"Update many operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_delete_one(self, filter_query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform delete_one operation"""
        try:
            self._debug_log(f"Deleting single document with filter: {filter_query}", "INFO")

            result = self.collection.delete_one(filter_query)

            self.document_count = result.deleted_count
            self._debug_log(f"Delete operation completed - Deleted: {result.deleted_count}", "INFO")

            return {
                "operation": "delete_one",
                "success": True,
                "deleted_count": result.deleted_count,
                "acknowledged": result.acknowledged,
                "document_count": result.deleted_count
            }

        except Exception as e:
            error_msg = f"Delete one operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _perform_delete_many(self, filter_query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform delete_many operation"""
        try:
            self._debug_log(f"Deleting multiple documents with filter: {filter_query}", "INFO")

            result = self.collection.delete_many(filter_query)

            self.document_count = result.deleted_count
            self._debug_log(f"Delete operation completed - Deleted: {result.deleted_count}", "INFO")

            return {
                "operation": "delete_many",
                "success": True,
                "deleted_count": result.deleted_count,
                "acknowledged": result.acknowledged,
                "document_count": result.deleted_count
            }

        except Exception as e:
            error_msg = f"Delete many operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def perform_operation(self) -> Data:
        """Main method to perform MongoDB operations"""
        try:
            start_time = time.time()

            self._debug_log(f"Starting MongoDB operation: {self.operation}", "INFO")

            # Connect to MongoDB
            self._connect_to_mongodb()

            # Extract data based on operation type
            if self.operation in ["insert_one", "insert_many"]:
                document_data = self._extract_data(self.document_data)
                if not document_data:
                    raise ValueError("Document data is required for insert operations")

            if self.operation in ["find_one", "find_many", "update_one", "update_many", "delete_one", "delete_many"]:
                filter_query = self._extract_data(self.filter_query)
                if not filter_query:
                    filter_query = {}  # Empty filter means all documents

            if self.operation in ["update_one", "update_many"]:
                update_data = self._extract_data(self.update_data)
                if not update_data:
                    raise ValueError("Update data is required for update operations")

            # Perform the requested operation
            if self.operation == "insert_one":
                if isinstance(document_data, list):
                    raise ValueError("insert_one expects a single document, not a list")
                result = self._perform_insert_one(document_data)

            elif self.operation == "insert_many":
                if isinstance(document_data, dict):
                    document_data = [document_data]  # Convert single doc to list
                if not isinstance(document_data, list):
                    raise ValueError("insert_many expects a list of documents")
                result = self._perform_insert_many(document_data)

            elif self.operation == "find_one":
                result = self._perform_find_one(filter_query)

            elif self.operation == "find_many":
                result = self._perform_find_many(filter_query)

            elif self.operation == "update_one":
                result = self._perform_update_one(filter_query, update_data)

            elif self.operation == "update_many":
                result = self._perform_update_many(filter_query, update_data)

            elif self.operation == "delete_one":
                result = self._perform_delete_one(filter_query)

            elif self.operation == "delete_many":
                result = self._perform_delete_many(filter_query)

            else:
                raise ValueError(f"Unsupported operation: {self.operation}")

            # Store results
            self.operation_result = result
            self.operation_success = True

            elapsed_time = time.time() - start_time

            # Add metadata to result
            result.update({
                "collection_name": self.collection_name,
                "database_name": self.database_name,
                "processing_time_seconds": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

            self._debug_log(f"✅ MongoDB operation completed successfully in {elapsed_time:.3f}s", "INFO")
            return Data(data=result)

        except Exception as e:
            # Store failure state
            self.operation_success = False
            self.error_details = str(e)

            error_msg = f"MongoDB operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")

            # Return error information
            error_result = {
                "operation": self.operation,
                "success": False,
                "error": error_msg,
                "collection_name": self.collection_name,
                "database_name": self.database_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            return Data(data=error_result)

        finally:
            # Clean up connection
            if self.client:
                try:
                    self.client.close()
                    self._debug_log("MongoDB connection closed", "INFO")
                except:
                    pass  # Ignore cleanup errors

    def get_success(self) -> Data:
        """Get operation success status"""
        return Data(data={"success": self.operation_success})

    def get_document_count(self) -> Data:
        """Get the number of documents affected by the operation"""
        return Data(data={"document_count": self.document_count})

    def get_error_details(self) -> Data:
        """Get detailed error information"""
        if not self.operation_success and self.error_details:
            return Data(data={"error": self.error_details})
        return Data(data={"error": None})

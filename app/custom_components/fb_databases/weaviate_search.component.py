import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from langflow.custom import Component
from langflow.inputs import StrInput, SecretStrInput, FloatInput, IntInput, BoolInput, DropdownInput, MessageTextInput
from langflow.io import Output
from langflow.schema import Data

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.filters import Filter
from weaviate.collections.classes.grpc import HybridFusion


class EmbeddingModel(str, Enum):
    """Available embedding models"""
    OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class WeaviateSearchComponent(Component):
    display_name = "Weaviate Vector Search"
    description = "Search for documents in Weaviate using vector similarity with configurable embedding models"
    documentation: str = "Connect to Weaviate and perform semantic search using various embedding models"
    icon = "database-zap"
    name = "WeaviateSearch"
    
    # Class-level connection cache to reuse connections
    _connection_cache = {}

    inputs = [
        SecretStrInput(
            name="weaviate_url",
            display_name="Weaviate URL",
            info="Weaviate instance URL (e.g., http://localhost:8080 or Weaviate Cloud URL)",
            required=True,
        ),
        SecretStrInput(
            name="weaviate_api_key",
            display_name="Weaviate API Key",
            info="Weaviate API key (required for Weaviate Cloud)",
            required=False,
        ),
        DropdownInput(
            name="embedding_model",
            display_name="Embedding Model",
            info="Embedding model to use for vectorization",
            options=[model.value for model in EmbeddingModel],
            value=EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL.value,
            required=True,
        ),
        SecretStrInput(
            name="embedding_api_key",
            display_name="Embedding Model API Key",
            info="API key for the embedding model (e.g., OpenAI API key)",
            required=True,
        ),
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            info="Name of the Weaviate collection to search",
            required=True,
        ),
        MessageTextInput(
            name="search_query",
            display_name="Search Query",
            info="Text query to search for similar documents",
            required=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="filter_property_name",
            display_name="Filter Property Name",
            info="Name of the Weaviate property to filter by",
            required=True,
        ),
        MessageTextInput(
            name="filter_property_value",
            display_name="Filter Property Value",
            info="Name of the Weaviate property value to filter by",
            required=True,
        ),
        IntInput(
            name="top_k",
            display_name="Top K Results",
            info="Number of top results to return",
            value=5,
            required=True,
        ),
        FloatInput(
            name="similarity_threshold",
            display_name="Similarity Threshold",
            info="Minimum similarity score (0.0 to 1.0)",
            value=0.7,
            required=False,
        ),
        IntInput(
            name="connection_timeout",
            display_name="Connection Timeout (seconds)",
            info="Connection timeout in seconds",
            value=30,
            required=False,
        ),
        BoolInput(
            name="include_distances",
            display_name="Include Similarity Scores",
            info="Include similarity/distance scores in results",
            value=True,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Logging",
            info="Enable detailed logging",
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="Search Results", name="search_results", method="search_documents"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = None
        self.search_results_data = None
        self.search_success = False
        self.result_count = 0
        self.error_details = None
        
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
            import weaviate
            self._debug_log("Weaviate dependency check passed", "INFO")
        except ImportError:
            raise ImportError(
                "weaviate-client package is required. "
                "Please install it: pip install weaviate-client"
            )
        self._debug_log("Weaviate dependency check passed", "INFO")

    def _get_connection_key(self) -> str:
        """Generate a unique key for connection caching"""
        return f"{self.weaviate_url}:{self.weaviate_api_key}:{self.embedding_model}:{self.embedding_api_key}"
    
    def _connect_to_weaviate(self) -> bool:
        """Establish connection to Weaviate instance with connection reuse"""
        try:
            self._debug_log("Checking dependencies...", "INFO")
            self._check_dependencies()
            
            # Check if we have a cached connection
            connection_key = self._get_connection_key()
            
            if connection_key in self._connection_cache:
                cached_client = self._connection_cache[connection_key]
                try:
                    if cached_client.is_ready():
                        self.client = cached_client
                        self._debug_log("Reusing existing Weaviate connection", "INFO")
                        return True
                    else:
                        # Remove stale connection
                        self._debug_log("Removing stale cached connection", "INFO")
                        try:
                            cached_client.close()
                        except:
                            pass
                        del self._connection_cache[connection_key]
                except Exception as e:
                    self._debug_log(f"Cached connection invalid: {str(e)}", "WARNING")
                    try:
                        cached_client.close()
                    except:
                        pass
                    del self._connection_cache[connection_key]
            
            self._debug_log(f"Creating new connection to Weaviate at {self.weaviate_url}", "INFO")
            
            # Configure headers for embedding model API key
            headers = {}
            if self.embedding_model.startswith("text-embedding"):
                headers["X-OpenAI-Api-Key"] = self.embedding_api_key
            
            # Connect to Weaviate
            if self.weaviate_url.startswith("http://localhost") or self.weaviate_url.startswith("http://127.0.0.1"):
                # Local connection
                client = weaviate.connect_to_local(
                    host=self.weaviate_url.replace("http://", "").split(":")[0],
                    port=int(self.weaviate_url.split(":")[-1]) if ":" in self.weaviate_url else 8080,
                    headers=headers
                )
            else:
                # Cloud or remote connection
                auth_credentials = None
                if self.weaviate_api_key:
                    auth_credentials = Auth.api_key(self.weaviate_api_key)
                
                client = weaviate.connect_to_wcs(
                    cluster_url=self.weaviate_url,
                    auth_credentials=auth_credentials,
                    headers=headers
                )
            
            # Test connection
            if client.is_ready():
                self.client = client
                # Cache the connection for reuse
                self._connection_cache[connection_key] = client
                self._debug_log("Successfully connected to Weaviate and cached connection", "INFO")
                return True
            else:
                self._debug_log("Failed to connect to Weaviate", "ERROR")
                try:
                    client.close()
                except:
                    pass
                return False
                
        except Exception as e:
            self._debug_log(f"Error connecting to Weaviate: {str(e)}", "ERROR")
            return False

    def _perform_search(self) -> List[Dict[str, Any]]:
        """Perform the vector search in Weaviate"""
        try:
            self._debug_log(f"Searching collection '{self.collection_name}' for query: '{self.search_query}'", "INFO")
            
            # Get the collection
            collection = self.client.collections.get(self.collection_name)
            
            # Configure what to include in results
            metadata_config = None
            if self.include_distances :
                metadata_config = MetadataQuery(
                    score=self.include_distances,
                    explain_score=self.include_distances,
                    distance=self.include_distances,
                )
            
            # Execute the search with metadata parameter
            self._debug_log(f"Executing search with top_k={self.top_k}", "INFO")

            response = collection.query.hybrid(
                query=self.search_query,
                limit=self.top_k,
                alpha=0.75,
                max_vector_distance=0.7,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=metadata_config,
                filters=Filter.by_property(self.filter_property_name).equal(self.filter_property_value)
            )

            self._debug_log(f"Processing search results : {response.objects}", "INFO")
            results = []
            for obj in response.objects:
                result_item = {
                    "id": str(obj.uuid),
                    "properties": obj.properties,
                }

                # Add similarity scores if requested
                if self.include_distances and hasattr(obj, 'metadata') and obj.metadata:
                    if hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                        result_item["score"] = float(obj.metadata.score)

                self._debug_log(f"Result object: {result_item}", "ERROR")
                # Apply similarity threshold if set

                if self.similarity_threshold and result_item.get("score", 0.0) >= self.similarity_threshold:
                    results.append(result_item)
                elif not self.similarity_threshold:
                    results.append(result_item)

            self.result_count = len(results)
            self._debug_log(f"Found {self.result_count} results", "INFO")
            
            return results
            
        except Exception as e:
            error_msg = f"Search operation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _format_results_for_agent(self, results: List[Dict[str, Any]]) -> str:
        """Format search results in a concise way for agent consumption"""
        if not results:
            return f"No results found for query: '{self.search_query}'"
        
        formatted_lines = [f"Found {len(results)} results for query: '{self.search_query}'"]
        
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 for agent
            formatted_lines.append(f"\nResult {i}:")
            
            # Add key properties
            properties = result.get('properties', {})
            for key, value in properties.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                formatted_lines.append(f"  {key}: {value}")
            
            # Add distance if available
            if 'distance' in result:
                formatted_lines.append(f"  similarity_distance: {result['distance']:.4f}")
        
        return "\n".join(formatted_lines)

    @classmethod
    def cleanup_connections(cls):
        """Clean up all cached connections - call when shutting down"""
        for key, client in cls._connection_cache.items():
            try:
                client.close()
            except:
                pass
        cls._connection_cache.clear()

    def search_documents(self) -> Data:
        """Main method to perform vector search"""
        try:
            start_time = time.time()
            
            self._debug_log(f"Starting Weaviate search operation", "INFO")
            
            # Connect to Weaviate
            if not self._connect_to_weaviate():
                raise Exception("Failed to connect to Weaviate")
            
            # Perform the search
            search_results = self._perform_search()
            
            # Store results
            self.search_results_data = search_results
            self.search_success = True
            
            elapsed_time = time.time() - start_time
            
            # Prepare comprehensive response
            response_data = {
                "search_query": self.search_query,
                "collection_name": self.collection_name,
                "results": search_results,
                "result_count": self.result_count,
                "top_k_requested": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedding_model,
                "processing_time_seconds": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
            self._debug_log(f"✅ Search completed successfully in {elapsed_time:.3f}s", "INFO")
            
            # Create formatted text for agent tool mode
            formatted_text = self._format_results_for_agent(search_results)
            
            return Data(data=response_data, text=formatted_text)
            
        except Exception as e:
            # Store failure state
            self.search_success = False
            self.error_details = str(e)
            
            error_msg = f"Weaviate search failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            
            # Return error information
            error_result = {
                "search_query": self.search_query,
                "collection_name": self.collection_name,
                "success": False,
                "error": error_msg,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            return Data(data=error_result, text=error_msg)
            
        finally:
            # Clean up connection
            if self.client:
                try:
                    self.client.close()
                    self._debug_log("Weaviate connection closed", "INFO")
                except:
                    pass  # Ignore cleanup errors

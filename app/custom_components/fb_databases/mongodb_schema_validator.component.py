import logging
import time
from enum import Enum
from typing import Any, Dict, List

from langflow.custom import Component
from langflow.inputs import DataInput, BoolInput, DropdownInput
from langflow.io import Output
from langflow.schema import Data
from pydantic import BaseModel, Field, ValidationError


class CollectionName(str, Enum):
    """Available collection names for schema validation"""
    PROCESSED_VIDEOS = "processed_videos"


# Pydantic schemas for each collection
class SecurityAssessmentSchema(BaseModel):
    """Schema for security assessment within timeframe"""
    threat_level: str = Field(..., min_length=1)
    confidence: str = Field(..., min_length=1)
    specific_concerns: List[str] = Field(default_factory=list)
    normal_operations: List[str] = Field(default_factory=list)


class TimeframeSchema(BaseModel):
    """Schema for individual timeframe in processed timeline"""
    timeframe_start: str = Field(..., min_length=1)
    timeframe_end: str = Field(..., min_length=1)
    timeframe_video_description: str = Field(..., min_length=1)
    environmental_conditions: Dict[str, Any] = Field(default_factory=dict)
    security_assessment: SecurityAssessmentSchema
    continuity_notes: str = Field(..., min_length=1)


class ProcessedVideosSchema(BaseModel):
    """Schema for processed_videos collection"""
    video_id: str = Field(..., min_length=1)
    processed_timeline: List[TimeframeSchema] = Field(..., min_items=1)


class MongoDBSchemaValidatorComponent(Component):
    display_name = "MongoDB Schema Validator"
    description = "Validate data against MongoDB collection schemas using Pydantic"
    documentation: str = "Validate input data against predefined collection schemas before database operations"
    icon = "shield-check"
    name = "MongoDBSchemaValidator"

    inputs = [
        DropdownInput(
            name="collection_name",
            display_name="Collection Name",
            info="Target collection name for schema validation",
            options=[collection.value for collection in CollectionName],
            required=True,
        ),
        DataInput(
            name="input_data",
            display_name="Input Data",
            info="Data to validate against the collection schema",
            required=True,
        ),
        BoolInput(
            name="strict_validation",
            display_name="Strict Validation",
            info="Enable strict validation (extra fields not allowed)",
            value=True,
        ),
        BoolInput(
            name="auto_generate_timestamps",
            display_name="Auto Generate Timestamps",
            info="Automatically add created_at/updated_at timestamps",
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
        Output(display_name="Validated Data", name="validated_data", method="validate_data"),
        Output(display_name="Success", name="success", method="get_success"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validated_data = None
        self.validation_success = False
        self.validation_errors = []
        self.schema_used = None

        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Schema mapping
        self.schema_mapping = {
            CollectionName.PROCESSED_VIDEOS: ProcessedVideosSchema,
        }

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

    def _extract_data(self, input_data: Any) -> Dict[str, Any]:
        """Extract data from various input formats"""
        try:
            self._debug_log(f"Extracting data from input type: {type(input_data)}", "INFO")

            # Handle list containing Data objects (common from some components)
            if isinstance(input_data, list) and len(input_data) > 0:
                if hasattr(input_data[0], 'data'):
                    input_data = input_data[0]
                else:
                    data = input_data[0]
                    return data if isinstance(data, dict) else {"value": data}

            # Handle Data object
            if hasattr(input_data, 'data'):
                data = input_data.data
            else:
                data = input_data

            # Handle string JSON
            if isinstance(data, str):
                import json
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    raise ValueError("Input data must be valid JSON string or dictionary")

            # Ensure we have a dictionary
            if not isinstance(data, dict):
                raise ValueError("Input data must be a dictionary")

            return data

        except Exception as e:
            self._debug_log(f"Error extracting data: {str(e)}", "ERROR")
            raise

    def _add_timestamps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamps to data if auto_generate_timestamps is enabled"""
        if not self.auto_generate_timestamps:
            return data

        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # Add created_at if not present
        if "created_at" not in data:
            data["created_at"] = current_time

        # Add updated_at
        data["updated_at"] = current_time

        return data

    def _get_schema_class(self, collection_name: str) -> BaseModel:
        """Get the appropriate schema class for the collection"""
        try:
            collection_enum = CollectionName(collection_name)
            schema_class = self.schema_mapping.get(collection_enum)

            if not schema_class:
                raise ValueError(f"No schema defined for collection: {collection_name}")

            return schema_class

        except ValueError as e:
            if "is not a valid CollectionName" in str(e):
                available_collections = [col.value for col in CollectionName]
                raise ValueError(
                    f"Invalid collection name: {collection_name}. "
                    f"Available collections: {', '.join(available_collections)}"
                )
            raise

    def _validate_against_schema(self, data: Dict[str, Any], schema_class: BaseModel) -> BaseModel:
        """Validate data against the schema"""
        try:
            self._debug_log(f"Validating data against {schema_class.__name__}", "INFO")

            # Configure validation settings
            if self.strict_validation:
                # In strict mode, extra fields are not allowed
                validated_instance = schema_class.parse_obj(data)
            else:
                # In non-strict mode, allow extra fields
                validated_instance = schema_class.parse_obj(data)

            self._debug_log("Schema validation passed", "INFO")
            return validated_instance

        except ValidationError as e:
            self._debug_log(f"Schema validation failed: {str(e)}", "ERROR")
            # Store validation errors for detailed reporting
            self.validation_errors = []
            for error in e.errors():
                error_detail = {
                    "field": " -> ".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input")
                }
                self.validation_errors.append(error_detail)
            raise

        except Exception as e:
            self._debug_log(f"Unexpected validation error: {str(e)}", "ERROR")
            raise

    def validate_data(self) -> Data:
        """Main method to validate data against collection schema"""
        try:
            start_time = time.time()

            self._debug_log(f"Starting schema validation for collection: {self.collection_name}", "INFO")

            # Extract and prepare data
            raw_data = self._extract_data(self.input_data)
            self._debug_log(f"Extracted data with {len(raw_data)} fields", "INFO")
            self._debug_log(f"Data: {raw_data}", "INFO")

            # Add timestamps if configured
            processed_data = self._add_timestamps(raw_data.copy())
            self._debug_log(f"Processed data with timestamps: {processed_data}", "INFO")

            # Get appropriate schema class
            schema_class = self._get_schema_class(self.collection_name)
            self.schema_used = schema_class.__name__

            # Validate against schema
            validated_instance = self._validate_against_schema(processed_data, schema_class)

            # Convert back to dict for output
            self.validated_data = validated_instance.dict()
            self.validation_success = True

            elapsed_time = time.time() - start_time

            # Prepare comprehensive output
            validation_result = {
                "validated_data": self.validated_data,
                "validation_success": True,
                "collection_name": self.collection_name,
                "schema_used": self.schema_used,
                "fields_validated": len(self.validated_data),
                "strict_validation": self.strict_validation,
                "auto_timestamps": self.auto_generate_timestamps,
                "processing_time_seconds": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            self._debug_log(f"✅ Schema validation completed successfully in {elapsed_time:.3f}s", "INFO")
            return Data(data=validation_result)

        except Exception as e:
            # Store failure state
            self.validation_success = False

            error_msg = f"Schema validation failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")

            # Return detailed error information
            error_result = {
                "validation_success": False,
                "error": error_msg,
                "collection_name": self.collection_name,
                "schema_used": getattr(self, 'schema_used', None),
                "validation_errors": self.validation_errors,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            return Data(data=error_result)

    def get_success(self) -> Data:
        """Get validation success status"""
        return Data(data={"success": self.validation_success})

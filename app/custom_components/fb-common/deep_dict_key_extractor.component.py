import logging
import time
from typing import Any, Dict, List, Optional, Union

from langflow.custom import Component
from langflow.inputs import StrInput, DataInput, BoolInput, MessageTextInput
from langflow.io import Output
from langflow.schema import Data


class DeepKeyExtractorComponent(Component):
    display_name = "Deep Dict Key Extractor"
    description = "Extract deeply nested values from data using dot notation (e.g., 'key1.key2.key3')"
    documentation: str = "Extract values from nested dictionaries and lists using dot notation path"
    icon = "search"
    name = "DeepDictKeyExtractor"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Data",
            info="Data object to extract values from",
            required=True,
        ),
        MessageTextInput(
            name="key_path",
            display_name="Key Path",
            info="Dot-separated path to extract (e.g., 'response_data.video_analysis_segments.0.timeframe_start')",
            required=True,
        ),
        StrInput(
            name="default_value",
            display_name="Default Value",
            info="Default value to return if key path is not found (optional)",
            required=False,
        ),
        BoolInput(
            name="return_all_matches",
            display_name="Return All Matches",
            info="Return all matching values when path contains wildcards or arrays",
            value=False,
        ),
        BoolInput(
            name="case_sensitive",
            display_name="Case Sensitive",
            info="Enable case-sensitive key matching",
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
        Output(display_name="Extracted Value", name="extracted_value", method="extract_value"),
        Output(display_name="Success", name="success", method="get_success"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extracted_value = None
        self.extraction_success = False
        self.path_found = False
        self.path_steps = []
        self.error_message = None

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

    def _extract_data(self, input_data: Any) -> Union[Dict[str, Any], List[Any], Any]:
        """Extract data from various input formats"""
        try:
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
                    self._debug_log("Input is a string but not valid JSON, treating as raw string", "INFO")
                    return data

            return data

        except Exception as e:
            self._debug_log(f"Error extracting data: {str(e)}", "ERROR")
            raise

    def _parse_key_path(self, key_path: str) -> List[str]:
        """Parse dot-separated key path into individual keys"""
        if not key_path:
            return []

        # Split by dots but handle escaped dots
        keys = []
        current_key = ""
        i = 0

        while i < len(key_path):
            if key_path[i] == '.' and (i == 0 or key_path[i - 1] != '\\'):
                if current_key:
                    keys.append(current_key)
                    current_key = ""
            elif key_path[i] == '\\' and i + 1 < len(key_path) and key_path[i + 1] == '.':
                current_key += '.'
                i += 1  # Skip the escaped dot
            else:
                current_key += key_path[i]
            i += 1

        if current_key:
            keys.append(current_key)

        return keys

    def _is_array_index(self, key: str) -> tuple[bool, Optional[int]]:
        """Check if key is an array index and return the index"""
        try:
            index = int(key)
            return True, index
        except ValueError:
            return False, None

    def _get_key_from_dict(self, data: Dict[str, Any], key: str) -> tuple[bool, Any]:
        """Get value from dictionary with optional case-insensitive matching"""
        if self.case_sensitive:
            if key in data:
                return True, data[key]
        else:
            # Case-insensitive search
            for dict_key in data.keys():
                if isinstance(dict_key, str) and dict_key.lower() == key.lower():
                    return True, data[dict_key]

        return False, None

    def _extract_nested_value(self, data: Any, keys: List[str], current_path: str = "") -> tuple[bool, Any]:
        """Recursively extract value from nested data structure"""
        if not keys:
            return True, data

        current_key = keys[0]
        remaining_keys = keys[1:]
        new_path = f"{current_path}.{current_key}" if current_path else current_key

        self._debug_log(f"Processing key '{current_key}' at path '{new_path}'", "DEBUG")

        # Handle dictionary access
        if isinstance(data, dict):
            found, value = self._get_key_from_dict(data, current_key)
            if found:
                self.path_steps.append({
                    "step": len(self.path_steps) + 1,
                    "key": current_key,
                    "path": new_path,
                    "type": "dict",
                    "found": True
                })
                return self._extract_nested_value(value, remaining_keys, new_path)
            else:
                self.path_steps.append({
                    "step": len(self.path_steps) + 1,
                    "key": current_key,
                    "path": new_path,
                    "type": "dict",
                    "found": False,
                    "available_keys": list(data.keys()) if len(data) <= 20 else f"{len(data)} keys"
                })
                return False, None

        # Handle list/array access
        elif isinstance(data, list):
            is_index, index = self._is_array_index(current_key)
            if is_index:
                if 0 <= index < len(data):
                    self.path_steps.append({
                        "step": len(self.path_steps) + 1,
                        "key": current_key,
                        "path": new_path,
                        "type": "list",
                        "index": index,
                        "found": True
                    })
                    return self._extract_nested_value(data[index], remaining_keys, new_path)
                else:
                    self.path_steps.append({
                        "step": len(self.path_steps) + 1,
                        "key": current_key,
                        "path": new_path,
                        "type": "list",
                        "index": index,
                        "found": False,
                        "list_length": len(data)
                    })
                    return False, None
            else:
                # Try to find key in list of dictionaries
                if self.return_all_matches:
                    matches = []
                    for i, item in enumerate(data):
                        if isinstance(item, dict):
                            found, value = self._get_key_from_dict(item, current_key)
                            if found:
                                sub_found, sub_value = self._extract_nested_value(value, remaining_keys,
                                                                                  f"{new_path}[{i}]")
                                if sub_found:
                                    matches.append(sub_value)

                    if matches:
                        self.path_steps.append({
                            "step": len(self.path_steps) + 1,
                            "key": current_key,
                            "path": new_path,
                            "type": "list_search",
                            "found": True,
                            "matches_count": len(matches)
                        })
                        return True, matches

                self.path_steps.append({
                    "step": len(self.path_steps) + 1,
                    "key": current_key,
                    "path": new_path,
                    "type": "list_search",
                    "found": False
                })
                return False, None

        # Handle other data types
        else:
            self.path_steps.append({
                "step": len(self.path_steps) + 1,
                "key": current_key,
                "path": new_path,
                "type": type(data).__name__,
                "found": False,
                "error": f"Cannot access key '{current_key}' on {type(data).__name__}"
            })
            return False, None

    def extract_value(self) -> Data:
        """Main method to extract nested values"""
        try:
            start_time = time.time()

            self._debug_log(f"Starting deep key extraction with path: {self.key_path}", "INFO")

            # Extract and prepare data
            raw_data = self._extract_data(self.input_data)
            self._debug_log(f"Input data type: {type(raw_data).__name__}", "INFO")

            # Parse key path
            keys = self._parse_key_path(self.key_path)
            if not keys:
                raise ValueError("Key path cannot be empty")

            self._debug_log(f"Parsed key path into {len(keys)} steps: {keys}", "INFO")

            # Reset path tracking
            self.path_steps = []

            # Extract the value
            found, value = self._extract_nested_value(raw_data, keys)

            if found:
                self.extracted_value = value
                self.extraction_success = True
                self.path_found = True

                elapsed_time = time.time() - start_time

                # Prepare comprehensive output
                extraction_result = {
                    "extracted_value": self.extracted_value,
                    "extraction_success": True,
                    "key_path": self.key_path,
                    "processing_time_seconds": elapsed_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                self._debug_log(f"✅ Key extraction completed successfully in {elapsed_time:.3f}s", "INFO")
                return Data(data=extraction_result)

            else:
                # Value not found, use default if provided
                if hasattr(self, 'default_value') and self.default_value is not None:
                    self.extracted_value = self.default_value
                    self.extraction_success = True
                    self.path_found = False

                    self._debug_log(f"Key path not found, using default value: {self.default_value}", "INFO")

                    extraction_result = {
                        "extracted_value": self.extracted_value,
                        "extraction_success": True,
                        "key_path": self.key_path,
                        "path_steps": self.path_steps,
                        "used_default": True,
                        "default_value": self.default_value,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    return Data(data=extraction_result)

                else:
                    raise ValueError(f"Key path '{self.key_path}' not found in data structure")

        except Exception as e:
            # Store failure state
            self.extraction_success = False
            self.error_message = str(e)

            error_msg = f"Key extraction failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")

            # Return detailed error information
            error_result = {
                "extraction_success": False,
                "error": error_msg,
                "key_path": self.key_path,
                "path_steps": self.path_steps,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            return Data(data=error_result)

    def get_success(self) -> Data:
        """Get extraction success status"""
        return Data(data={"success": self.extraction_success})

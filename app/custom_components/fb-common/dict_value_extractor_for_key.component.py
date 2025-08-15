import json
import logging
from typing import Any

from langflow.custom import Component
from langflow.inputs import MessageTextInput, DataInput
from langflow.schema import Data
from langflow.schema.message import Message
from langflow.io import Output


class DictValueExtractorForKey(Component):
    display_name = "Dict Value Extractor for Key"
    description = "Extract value from dictionary for a given key and return as string for next component"
    icon = "file-json"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Value",
            info="Dictionary to extract value from",
            value="",
            tool_mode=True,
        ),
        MessageTextInput(
            name="key",
            display_name="Key",
            info="Key to extract value for",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Value for Key", name="value", method="get_value"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parsed_data = None

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

    def build(self) -> Any:
        try:
            # Handle escaped JSON strings
            json_string = self.input_value
            json_input = json_string.strip()
            if json_input.startswith('"') and json_input.endswith('"'):
                json_input = json_input[1:-1].replace('\\"', '"').replace('\\\\', '\\')

            # Parse JSON
            self.parsed_data = json.loads(json_input)
            self._debug_log(f"JSON parsed successfully {self.parsed_data}", "INFO")
            return self.parsed_data

        except json.JSONDecodeError as e:
            self._debug_log(f"Invalid JSON format: {str(e)}", "ERROR")
            raise

    def get_value(self) -> Message:
        # if not self.parsed_data:
        #     self.build()
        #
        value = self.input_data.data.get(self.key, "")

        return Message(
            text=str(value)
        )

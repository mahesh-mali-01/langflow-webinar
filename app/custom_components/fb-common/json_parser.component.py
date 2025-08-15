import json

from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.schema import Data


class SimpleJSONParser(Component):
    display_name = "Simple JSON Parser"
    description = "Parse JSON string into accessible object"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Input Value",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="parsed_data", name="parsed_data", method="build"),
    ]

    def build(self) -> Data:
        try:
            # Handle escaped JSON strings
            json_string = self.input_value
            json_input = json_string.strip()
            if json_input.startswith('"') and json_input.endswith('"'):
                json_input = json_input[1:-1].replace('\\"', '"').replace('\\\\', '\\')

            # Parse JSON
            parsed_data = json.loads(json_input)

            return Data(
                data=parsed_data
            )

        except json.JSONDecodeError as e:
            error_data = {
                "error": f"Invalid JSON: {str(e)}",
                "raw_input": json_string
            }
            return Data(
                data=error_data,
                text=f"JSON Parse Error: {str(e)}"
            )

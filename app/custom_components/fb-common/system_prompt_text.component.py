from langflow.custom import Component
from langflow.io import MultilineInput, Output
from langflow.schema.message import Message


class SystemPromptTextComponent(Component):
    display_name = "SYSTEM_PROMPT_TEXT"
    description = "Get text inputs from the Playground."
    icon = "type"
    name = "TextInput"

    inputs = [
        MultilineInput(
            name="system_prompt_text",
            display_name="system_prompt",
            info="Text to be passed as input.",
        ),
    ]
    outputs = [
        Output(display_name="Message", name="system_prompt", method="text_response"),
    ]

    def text_response(self) -> Message:
        return Message(
            text=self.system_prompt_text,
        )

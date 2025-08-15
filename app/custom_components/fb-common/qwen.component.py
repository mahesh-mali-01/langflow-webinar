"""
Simplified Qwen VL Custom Component for Langflow
Asset Inspection Platform Integration

This is a simplified version to avoid compatibility issues.
"""

import requests

from langflow.custom import Component
from langflow.io import MessageTextInput, Output, IntInput, FloatInput
from langflow.schema import Data


class QwenVLSimpleComponent(Component):
    """
    Simplified Qwen VL component for image analysis via URL
    Configurable system and user prompts
    """

    display_name = "Qwen VL Simple"
    description = "Qwen Vision-Language model with configurable prompts and image URL input"
    icon = "eye"
    name = "QwenVLSimple"

    inputs = [
        MessageTextInput(
            name="base_url",
            display_name="Server URL",
            info="vLLM server URL",
            value="http://localhost:18000/v1"
        ),

        MessageTextInput(
            name="model_name",
            display_name="Model Name",
            info="Model name in vLLM",
            value="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
        ),

        MessageTextInput(
            name="image_url",
            display_name="Image URL",
            info="URL or file path to image for analysis",
            value=""
        ),

        MessageTextInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System prompt to define the AI's role and expertise",
            value="You are a specialized oil and gas asset inspection expert with deep knowledge of wellhead equipment, pipeline infrastructure, storage systems, power infrastructure, drilling equipment, and control systems. Your expertise includes component recognition, failure analysis, corrosion mechanisms, and industry standards."
        ),

        MessageTextInput(
            name="user_prompt",
            display_name="User Prompt",
            info="Specific analysis instructions for the image",
            value="Analyze this industrial facility image. Provide: 1. SCENE OVERVIEW: What type of facility and equipment is visible 2. COMPONENT CONDITION: Assess each visible component's condition (1-10 scale) 3. ISSUES IDENTIFIED: Any problems, damage, or safety concerns 4. RECOMMENDATIONS: Immediate actions and maintenance needs. Be specific and technical in your analysis."
        ),

        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum response tokens",
            value=8192
        ),

        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Response randomness (0.0-1.0)",
            value=0.1
        )
    ]

    outputs = [
        Output(display_name="Analysis", name="analysis", method="analyze_image")
    ]

    def analyze_image(self) -> Data:
        """Analyze the image from URL"""

        try:
            # Validate inputs
            if not self.image_url:
                return Data(
                    data={"error": "No image URL provided"},
                    text="Error: No image URL provided"
                )

            if not self.system_prompt:
                return Data(
                    data={"error": "System prompt is required"},
                    text="Error: System prompt is required"
                )

            if not self.user_prompt:
                return Data(
                    data={"error": "User prompt is required"},
                    text="Error: User prompt is required"
                )

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": self.image_url}
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Make API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300
            )

            if response.status_code != 200:
                return Data(
                    data={"error": f"API call failed: {response.status_code} - {response.text}"},
                    text=f"Error: API call failed with status {response.status_code}"
                )

            result = response.json()
            analysis = result["choices"][0]["message"]["content"]

            # Get additional metadata
            usage_info = result.get("usage", {})

            return Data(
                data={
                    "analysis": analysis,
                    "model": self.model_name,
                    "tokens_used": usage_info.get("total_tokens", 0),
                    "prompt_tokens": usage_info.get("prompt_tokens", 0),
                    "completion_tokens": usage_info.get("completion_tokens", 0),
                    "finish_reason": result["choices"][0].get("finish_reason", "unknown"),
                    "success": True,
                    "image_url": self.image_url
                },
                text=analysis
            )

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            return Data(
                data={"error": error_msg, "success": False},
                text=error_msg
            )

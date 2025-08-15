import json
import logging
import time
from typing import Any, Dict

import google.generativeai as genai
from langflow.custom import Component
from langflow.inputs import StrInput, SecretStrInput, FloatInput, IntInput, BoolInput, DataInput, \
    MultilineInput, MessageTextInput
from langflow.io import Output
from langflow.schema import Data


class GeminiVideoAnalyzerComponent(Component):
    display_name = "Gemini Video Analyzer"
    description = "Analyze videos using Google Gemini's video understanding capabilities"
    documentation: str = "Analyze uploaded videos with custom prompts using Google Gemini AI"
    icon = "video"
    name = "GeminiVideoAnalyzer"

    inputs = [
        MultilineInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System instructions for the video analysis",
            tool_mode=True,
            value="You are an expert video analyst. Analyze the provided video content thoroughly and provide detailed insights.",
            required=False,
        ),
        MultilineInput(
            name="user_prompt",
            display_name="User Prompt",
            info="Specific question or task for video analysis",
            tool_mode=True,
            value="Please analyze this video and describe what you see in detail.",
            required=True,
        ),
        MessageTextInput(
            name="video_id",
            display_name="Video ID",
            info="video id for analysis",
            value="",
            required=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="session id for analysis",
            value="",
            required=False,
        ),
        SecretStrInput(
            name="gemini_api_key",
            display_name="Gemini API Key",
            info="Your Google Gemini API key",
            required=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            info="Gemini model to use for analysis",
            value="gemini-2.5-pro",
            required=True,
        ),
        DataInput(
            name="file_data",
            display_name="File Data",
            info="File data from Gemini File Upload component",
            required=True,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Controls randomness in response generation (0.0-2.0)",
            value=0.1,
            required=False,
        ),
        IntInput(
            name="max_output_tokens",
            display_name="Max Output Tokens",
            info="Maximum number of tokens in the response",
            value=64000,
            required=False,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="Controls diversity of response (0.0-1.0)",
            value=0.95,
            required=False,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            info="Limits token selection to top K choices",
            value=40,
            required=False,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Logging",
            info="Enable detailed logging",
            value=True,
        ),
        BoolInput(
            name="stream_response",
            display_name="Stream Response",
            info="Enable streaming response for real-time output",
            value=False,
        ),
        BoolInput(
            name="use_json_schema",
            display_name="Use JSON Schema",
            info="Enable structured JSON output with predefined schema",
            value=True,
        ),
        MultilineInput(
            name="json_schema",
            display_name="JSON Schema",
            info="Custom JSON schema for structured output (optional)",
            tool_mode=True,
            value="",
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Analysis Result", name="analysis_result", method="analyze_video"),
        Output(display_name="Success", name="success", method="get_success"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_response = None
        self.response_text = None
        self.model_info = None
        self.analysis_success = False

        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            # Set default level, will be updated in _setup_logging
            self.logger.setLevel(logging.INFO)

    def _setup_logging(self):
        """Set up logging level based on verbose setting"""
        if hasattr(self, 'verbose'):
            self.logger.setLevel(logging.INFO if self.verbose else logging.ERROR)

    def _debug_log(self, message: str, level: str = "ERROR"):
        """Custom logging function with configurable log level"""
        # Ensure logging is set up properly
        self._setup_logging()

        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        log_level = level_mapping.get(level.upper(), logging.ERROR)
        self.logger.log(log_level, message)

        # Also log to Langflow's logger for visibility in UI
        if level in ["ERROR", "CRITICAL"] or (hasattr(self, 'verbose') and self.verbose):
            self.log(f"[{level}] {message}")

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        if genai is None:
            raise ImportError(
                "google-generativeai package is required. "
                "Please install it: pip install google-generativeai"
            )

    def _configure_genai(self):
        """Configure the Gemini AI client"""
        try:
            self._debug_log("Checking dependencies...", "INFO")
            self._check_dependencies()
            self._debug_log("Configuring Gemini AI for video analysis...", "INFO")
            genai.configure(api_key=self.gemini_api_key)
            self._debug_log("Gemini AI configured successfully", "INFO")
        except Exception as e:
            error_msg = f"Error configuring Gemini AI: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _validate_file_data(self, file_data: Any) -> Dict[str, Any]:
        """Validate and extract file information from input data"""
        try:
            self._debug_log("Validating file data input...", "INFO")

            # Handle Data object
            if hasattr(file_data, 'data'):
                data = file_data.data
            else:
                data = file_data

            # Ensure we have a dictionary
            if isinstance(data, str):
                import json
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    raise ValueError("File data must be a valid JSON string or dictionary")

            if not isinstance(data, dict):
                raise ValueError("File data must be a dictionary")

            # Validate required fields
            required_fields = ['name', 'uri', 'state']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field in file data: {field}")

            # Check if file is ready for analysis
            if data.get('state') != 'ACTIVE':
                raise ValueError(f"File is not ready for analysis. Current state: {data.get('state')}")

            if not data.get('upload_success', False):
                raise ValueError("File upload was not successful")

            self._debug_log(f"File validation passed: {data.get('name')} ({data.get('mime_type', 'unknown')})", "INFO")
            return data

        except Exception as e:
            self._debug_log(f"File data validation failed: {str(e)}", "ERROR")
            raise

    def _create_file_reference(self, file_data: Dict[str, Any]) -> Any:
        """Create a file reference object for Gemini API"""
        try:
            self._debug_log("Creating file reference for Gemini API...", "INFO")

            # Get the file using its name
            file_name = file_data['name']
            file_obj = genai.get_file(file_name)

            self._debug_log(f"File reference created: {file_obj.name}, State: {file_obj.state.name}", "INFO")

            # Double-check the file is still active
            if file_obj.state.name != "ACTIVE":
                raise ValueError(f"File is not in ACTIVE state: {file_obj.state.name}")

            return file_obj

        except Exception as e:
            error_msg = f"Error creating file reference: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _get_default_schema(self) -> dict:
        """Get the default JSON schema for video analysis"""
        return {
            "type": "object",
            "properties": {
                "video_analysis_segments": {
                    "type": "array",
                    "description": "Array of video analysis segments/events",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timeframe_start": {
                                "type": "string",
                                "description": "Time in MM:SS format"
                            },
                            "timeframe_end": {
                                "type": "string",
                                "description": "Time in MM:SS format"
                            },
                            "timeframe_video_description": {
                                "type": "string",
                                "description": "Detailed real-time description of all observable activities"
                            },
                            "environmental_conditions": {
                                "type": "object",
                                "properties": {
                                    "weather_visibility": {"type": "string"},
                                    "facility_status": {"type": "string"}
                                },
                                "required": ["weather_visibility", "facility_status"]
                            },
                            "security_assessment": {
                                "type": "object",
                                "properties": {
                                    "threat_level": {
                                        "type": "string",
                                        "enum": ["CRITICAL", "HIGH", "MODERATE", "LOW"]
                                    },
                                    "confidence": {
                                        "type": "string",
                                        "enum": ["HIGH", "MEDIUM", "LOW"]
                                    },
                                    "specific_concerns": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "normal_operations": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["threat_level", "confidence", "specific_concerns",
                                             "normal_operations"]
                            },
                            "continuity_notes": {
                                "type": "string",
                                "description": "How this segment relates to previous intervals"
                            }
                        },
                        "required": ["timeframe_start", "timeframe_end", "timeframe_video_description",
                                     "environmental_conditions", "security_assessment", "continuity_notes"]
                    }
                }
            },
            "required": ["video_analysis_segments"]
        }

    def _create_generation_config(self) -> genai.GenerationConfig:
        """Create generation configuration"""
        config_params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

        # Add optional parameters if provided
        if hasattr(self, 'top_p') and self.top_p is not None:
            config_params["top_p"] = self.top_p
        if hasattr(self, 'top_k') and self.top_k is not None:
            config_params["top_k"] = self.top_k

        # Add JSON schema configuration if enabled
        if hasattr(self, 'use_json_schema') and self.use_json_schema:
            config_params["response_mime_type"] = "application/json"

            # Use custom schema if provided, otherwise use default
            if hasattr(self, 'json_schema') and self.json_schema and self.json_schema.strip():
                try:
                    custom_schema = json.loads(self.json_schema)
                    config_params["response_schema"] = custom_schema
                    self._debug_log("Using custom JSON schema", "INFO")
                except json.JSONDecodeError as e:
                    self._debug_log(f"Invalid custom JSON schema, using default: {str(e)}", "WARN")
                    config_params["response_schema"] = self._get_default_schema()
            else:
                config_params["response_schema"] = self._get_default_schema()
                self._debug_log("Using default video analysis JSON schema", "INFO")

        self._debug_log(f"Generation config: {config_params}", "INFO")
        return genai.GenerationConfig(**config_params)

    def _analyze_video_content(self, file_obj: Any, prompt: str) -> Any:
        """Analyze video using Gemini API"""
        try:
            self._debug_log(f"Starting video analysis with model: {self.model_name}", "INFO")

            # Create the model with system instruction if provided
            model_kwargs = {"model_name": self.model_name}

            if hasattr(self, 'system_prompt') and self.system_prompt:
                model_kwargs["system_instruction"] = self.system_prompt
                self._debug_log(f"Using system instruction: {self.system_prompt[:100]}...", "INFO")

            model = genai.GenerativeModel(**model_kwargs)

            # Create generation config
            generation_config = self._create_generation_config()

            # Prepare the content parts (user prompt + video file)
            content_parts = [prompt, file_obj]

            self._debug_log(f"Generating content with {len(content_parts)} parts (user prompt + video)", "INFO")

            # Generate content
            if self.stream_response:
                self._debug_log("Using streaming response", "INFO")
                response = model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    stream=True
                )

                # Collect streaming response
                full_text = ""
                for chunk in response:
                    if chunk.text:
                        full_text += chunk.text

                # Create a response-like object for consistency
                class StreamResponse:
                    def __init__(self, text, usage_metadata=None):
                        self.text = text
                        self.usage_metadata = usage_metadata

                # Get usage metadata from the last chunk if available
                usage_metadata = getattr(response, 'usage_metadata', None)
                response = StreamResponse(full_text, usage_metadata)

            else:
                self._debug_log("Using standard response", "INFO")
                response = model.generate_content(
                    content_parts,
                    generation_config=generation_config
                )

            self._debug_log("Video analysis completed successfully", "INFO")
            return response

        except Exception as e:
            error_msg = f"Error during video analysis: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _get_response_data(self) -> Any:
        """Get the analysis response text"""
        try:
            if self.response_text and self.analysis_success:
                self._debug_log("hey mahesh Extracting JSON from response text", "INFO")
                response_data = json.loads(self.response_text)
                self._debug_log(f"JSON extracted: {response_data}", "INFO")
                return response_data
            return self.response_text
        except ImportError:
            raise Exception("JSON module is required for this operation")

    def analyze_video(self) -> Data:
        """Main method to analyze video content"""
        try:
            start_time = time.time()

            # Configure Gemini AI
            self._configure_genai()

            # Validate file data
            file_data = self._validate_file_data(self.file_data)

            # Create file reference
            file_obj = self._create_file_reference(file_data)

            # Analyze video
            response = self._analyze_video_content(file_obj, self.user_prompt)

            # Store results
            self.analysis_response = response
            self.response_text = response.text
            self.analysis_success = True

            # Store model info
            self.model_info = {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": getattr(self, 'top_p', None),
                "top_k": getattr(self, 'top_k', None),
                "stream_response": self.stream_response,
            }

            elapsed_time = time.time() - start_time

            # Prepare comprehensive output
            analysis_result = {
                "response_data": self._get_response_data(),
                "analysis_success": True,
                "processing_time_seconds": elapsed_time,
                "model_used": self.model_name,
                "file_analyzed": {
                    "name": file_data.get('name'),
                    "uri": file_data.get('uri'),
                    "mime_type": file_data.get('mime_type'),
                    "size_bytes": file_data.get('size_bytes'),
                },
                "generation_config": self.model_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            self._debug_log(f"🎉 Video analysis completed successfully in {elapsed_time:.2f}s!", "INFO")
            self._debug_log(f"Video analysis result: {analysis_result}", "INFO")

            # Return analysis result
            output_result = {
                "video_id": self.video_id,
                "session_id": self.session_id,
                "processed_timeline": analysis_result.get("response_data", {}).get("video_analysis_segments")
            }
            return Data(data=output_result)

        except Exception as e:
            # Store failure state
            self.analysis_success = False

            error_msg = f"Failed to analyze video: {str(e)}"
            self._debug_log(error_msg, "ERROR")

            # Return error information
            error_result = {
                "analysis_success": False,
                "error": error_msg,
                "model_used": getattr(self, 'model_name', 'unknown'),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            return Data(data=error_result)

    def get_success(self) -> Data:
        """Get analysis success status"""
        return Data(data={"success": self.analysis_success})

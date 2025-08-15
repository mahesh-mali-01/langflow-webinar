import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from langflow.custom import Component
from langflow.inputs import StrInput, SecretStrInput, IntInput, BoolInput, MessageTextInput
from langflow.io import Output
from langflow.schema import Data


class GeminiFileUploadComponent(Component):
    display_name = "Gemini File Upload & Poll"
    description = "Upload video/image files to Gemini and poll until processing is complete"
    documentation: str = "Upload files to Google Gemini and wait for processing completion"
    icon = "cloud-upload"
    name = "GeminiFileUploader"

    inputs = [
        MessageTextInput(
            name="file_path",
            display_name="File Path",
            info="Path to the video/image file to upload",
            required=True,
        ),
        SecretStrInput(
            name="gemini_api_key",
            display_name="Gemini API Key",
            info="Your Google Gemini API key",
            required=True,
        ),
        StrInput(
            name="mime_type",
            display_name="MIME Type",
            info="MIME type of the file (auto-detected if empty)",
            required=True,
            value="video/mp4",
        ),
        MessageTextInput(
            name="display_name",
            display_name="Display Name",
            info="Display name for the uploaded file (optional)",
            required=False,
        ),
        IntInput(
            name="polling_interval",
            display_name="Polling Interval (seconds)",
            info="Time to wait between polling requests",
            value=10,
            required=True,
        ),
        IntInput(
            name="max_wait_time",
            display_name="Max Wait Time (seconds)",
            info="Maximum time to wait for processing (0 = no limit)",
            value=600,  # 10 minutes default
            required=True,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Logging",
            info="Enable detailed logging",
            value=True,
        ),
        BoolInput(
            name="auto_detect_mime",
            display_name="Auto-detect MIME Type",
            info="Automatically detect MIME type from file extension",
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="File Info", name="file_info", method="upload_and_poll_file"),
        Output(display_name="Success", name="success", method="get_success"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uploaded_file = None
        self.final_status = None
        self.upload_success = False

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
            self._debug_log("Configuring Gemini AI...", "INFO")
            genai.configure(api_key=self.gemini_api_key)
            self._debug_log("Gemini AI configured successfully", "INFO")
        except Exception as e:
            error_msg = f"Error configuring Gemini AI: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _validate_file_path(self, file_path: str) -> bool:
        """Validate that the file exists and is readable"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"File is not readable: {file_path}")

            # Check file size (Gemini has limits)
            file_size = path.stat().st_size
            max_size = 5 * 1024 * 1024 * 1024  # 5GB limit
            if file_size > max_size:
                raise ValueError(f"File too large: {file_size / (1024 ** 3):.2f}GB (max: 2GB)")

            self._debug_log(f"File validation passed: {file_path} ({file_size / (1024 ** 2):.2f}MB)", "INFO")
            return True
        except Exception as e:
            self._debug_log(f"File validation failed: {str(e)}", "ERROR")
            raise

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type from file extension"""
        if self.mime_type and not self.auto_detect_mime:
            return self.mime_type

        # Auto-detect MIME type
        detected_mime, _ = mimetypes.guess_type(file_path)

        if detected_mime:
            self._debug_log(f"Auto-detected MIME type: {detected_mime}", "INFO")
            return detected_mime

        # Fallback to provided mime_type or default
        if self.mime_type:
            return self.mime_type

        # Default based on file extension
        ext = Path(file_path).suffix.lower()
        mime_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/avi',
            '.mov': 'video/quicktime',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
        }

        return mime_map.get(ext, 'application/octet-stream')

    def _get_display_name(self, file_path: str) -> str:
        """Get display name for the file"""
        if self.display_name:
            return self.display_name
        return Path(file_path).name

    def _upload_file(self, file_path: str) -> Any:
        """Upload file to Gemini"""
        try:
            self._debug_log(f"Starting upload for file: {file_path}", "INFO")

            # Validate file before upload
            self._validate_file_path(file_path)

            # Get MIME type and display name
            mime_type = self._detect_mime_type(file_path)
            display_name = self._get_display_name(file_path)

            self._debug_log(f"Upload parameters: File={file_path}, MIME={mime_type}, Display={display_name}", "INFO")

            # Upload file
            upload_response = genai.upload_file(
                path=file_path,
                mime_type=mime_type,
                display_name=display_name
            )

            self._debug_log(
                f"File uploaded successfully: Name={upload_response.name}, URI={upload_response.uri}, State={upload_response.state.name}, Size={getattr(upload_response, 'size_bytes', 'Unknown')} bytes",
                "INFO")

            return upload_response

        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def _poll_file_status(self, file_name: str) -> Any:
        """Poll file status until processing is complete"""
        try:
            start_time = time.time()
            poll_count = 0

            self._debug_log(f"Starting polling for file: {file_name}", "INFO")

            while True:
                poll_count += 1

                # Get current file status
                file_info = genai.get_file(file_name)

                self._debug_log(f"Poll #{poll_count} - Current state: {file_info.state.name}", "INFO")

                # Check if processing is complete
                if file_info.state.name == "ACTIVE":
                    elapsed_time = time.time() - start_time
                    self._debug_log(f"✅ File processing completed successfully in {elapsed_time:.1f} seconds!", "INFO")
                    return file_info

                elif file_info.state.name == "FAILED":
                    raise Exception(f"File processing failed for {file_name}")

                elif file_info.state.name == "PROCESSING":
                    elapsed = time.time() - start_time
                    self._debug_log(f"File still processing... (elapsed: {elapsed:.1f}s)", "INFO")
                else:
                    self._debug_log(f"Unexpected file state: {file_info.state.name}", "WARN")

                # Check timeout
                if self.max_wait_time > 0:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.max_wait_time:
                        raise TimeoutError(
                            f"File processing timeout after {elapsed_time:.1f} seconds "
                            f"(max: {self.max_wait_time}s)"
                        )

                # Wait before next poll
                time.sleep(self.polling_interval)

        except Exception as e:
            error_msg = f"Error polling file status: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            raise Exception(error_msg)

    def upload_and_poll_file(self) -> Data:
        """Main method to upload file and poll until ready"""
        try:
            # Configure Gemini AI
            self._configure_genai()

            # Upload file
            uploaded_file = self._upload_file(self.file_path)

            # Poll until processing is complete
            final_file = self._poll_file_status(uploaded_file.name)

            # Store results for other outputs
            self.uploaded_file = final_file
            self.final_status = final_file.state
            self.upload_success = True

            # Prepare comprehensive output data
            file_info = {
                "name": final_file.name,
                "uri": final_file.uri,
                "display_name": final_file.display_name,
                "mime_type": final_file.mime_type,
                "size_bytes": getattr(final_file, 'size_bytes', None),
                "state": final_file.state.name,
                "create_time": str(final_file.create_time),
                "update_time": str(final_file.update_time),
                "sha256_hash": getattr(final_file, 'sha256_hash', None),
                "upload_success": True,
                "processing_time_seconds": getattr(self, '_processing_time', None),
                "original_file_path": self.file_path,
            }

            self._debug_log(f"🎉 Upload and polling completed successfully!", "INFO")

            return Data(data=file_info)

        except Exception as e:
            # Store failure state
            self.upload_success = False
            self.final_status = "FAILED"

            error_msg = f"Failed to upload and poll file: {str(e)}"
            self._debug_log(error_msg, "ERROR")

            # Return error information
            error_info = {
                "upload_success": False,
                "error": error_msg,
                "original_file_path": self.file_path,
                "state": "FAILED"
            }

            return Data(data=error_info)

    def get_success(self) -> Data:
        """Get upload success status"""
        return Data(data={"success": self.upload_success})

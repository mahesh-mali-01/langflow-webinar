import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from langflow.custom import Component
from langflow.inputs import MessageTextInput, SecretStrInput, StrInput, BoolInput, IntInput
from langflow.io import Output
from langflow.schema import Data
from langflow.schema.message import Message


class S3VideoDownloader(Component):
    display_name = "S3 Video Downloader"
    description = "Download video files from S3 to local storage for processing"
    documentation = """
    This component downloads video files from Amazon S3 to local storage for processing in Langflow.
    It supports various video formats, progress tracking, and automatic cleanup configuration.
    The component validates S3 URLs, handles authentication, and provides detailed logging.
    """
    icon = "cloud-download"
    name = "S3VideoDownloader"

    inputs = [
        MessageTextInput(
            name="s3_url",
            display_name="S3 URL",
            info="Full S3 URL of the video file (e.g., https://bucket.s3.region.amazonaws.com/path/to/video.mp4)",
            required=True,
            tool_mode=True,
        ),
        SecretStrInput(
            name="aws_access_key_id",
            display_name="AWS Access Key ID",
            info="AWS Access Key ID for S3 authentication",
            required=True,
        ),
        SecretStrInput(
            name="aws_secret_access_key",
            display_name="AWS Secret Access Key",
            info="AWS Secret Access Key for S3 authentication",
            required=True,
        ),
        StrInput(
            name="aws_region",
            display_name="AWS Region",
            info="AWS region where the S3 bucket is located",
            value="us-east-1",
            required=True,
        ),
        StrInput(
            name="local_storage_path",
            display_name="Local Storage Path",
            info="Local directory path where video will be downloaded",
            value="/app/downloads/videos",
            required=True,
        ),
        BoolInput(
            name="create_subdirectory",
            display_name="Create Subdirectory",
            info="Create a subdirectory using timestamp for organization",
            value=True,
        ),
        BoolInput(
            name="validate_video_format",
            display_name="Validate Video Format",
            info="Validate that the file is a supported video format",
            value=True,
        ),
        IntInput(
            name="timeout_seconds",
            display_name="Download Timeout (seconds)",
            info="Timeout for download operation in seconds",
            value=300,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Logging",
            info="Enable detailed logging of download progress",
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="download_video")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.operation_success = False
        self.error_details = None
        self.local_file_path = None
        self.download_info = {}
        
        # Supported video formats
        self.supported_formats = {
            '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'
        }
        
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

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            self._debug_log("✅ Boto3 dependency check passed", "INFO")
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 operations. "
                "Please install it: pip install boto3"
            )

    def _validate_s3_url(self, s3_url: str) -> Dict[str, str]:
        """Validate and parse S3 URL"""
        try:
            parsed = urlparse(s3_url)
            
            # Check if it's a valid S3 URL
            if not parsed.scheme in ['https', 'http']:
                raise ValueError("S3 URL must use http or https protocol")
            
            # Extract bucket and key from different S3 URL formats
            if 's3.amazonaws.com' in parsed.netloc:
                # Format: https://bucket.s3.amazonaws.com/path/to/file
                bucket_name = parsed.netloc.split('.')[0]
                object_key = parsed.path.lstrip('/')
            elif 's3.' in parsed.netloc:
                # Format: https://bucket.s3.region.amazonaws.com/path/to/file
                bucket_name = parsed.netloc.split('.')[0]
                object_key = parsed.path.lstrip('/')
            elif parsed.netloc.endswith('.amazonaws.com'):
                # Format: https://s3.region.amazonaws.com/bucket/path/to/file
                #https://auvsi-demo-video-keyframes.s3.us-east-1.amazonaws.com/verkos-assets/security_videos/Friant_Roulette_Speed_Chaos__on_Fresno_Wild_Roads.mp4
                path_parts = parsed.path.lstrip('/').split('/')
                bucket_name = path_parts[0]
                object_key = '/'.join(path_parts[1:])
            else:
                raise ValueError("Invalid S3 URL format")
            
            if not bucket_name or not object_key:
                raise ValueError("Could not extract bucket name or object key from S3 URL")
            
            return {
                'bucket': bucket_name,
                'key': object_key,
                'filename': Path(object_key).name
            }
            
        except Exception as e:
            raise ValueError(f"Invalid S3 URL format: {str(e)}")

    def _validate_video_format(self, filename: str) -> bool:
        """Validate that the file is a supported video format"""
        if not self.validate_video_format:
            return True
        
        file_extension = Path(filename).suffix.lower()
        return file_extension in self.supported_formats

    def _create_local_directory(self, base_path: str) -> str:
        """Create local directory for video storage"""
        try:
            base_dir = Path(base_path)
            
            if self.create_subdirectory:
                # Create subdirectory with timestamp
                timestamp = int(time.time())
                storage_dir = base_dir / f"video_{timestamp}"
            else:
                storage_dir = base_dir
            
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if directory is writable
            if not os.access(storage_dir, os.W_OK):
                raise PermissionError(f"No write permission for directory: {storage_dir}")
            
            return str(storage_dir)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create local directory: {str(e)}")

    def _create_s3_client(self):
        """Create and configure S3 client"""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            # Create S3 client with credentials
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Test credentials by listing buckets (minimal operation)
            try:
                s3_client.list_buckets()
                self._debug_log("✅ S3 credentials validated", "INFO")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'InvalidAccessKeyId':
                    raise ValueError("Invalid AWS Access Key ID")
                elif error_code == 'SignatureDoesNotMatch':
                    raise ValueError("Invalid AWS Secret Access Key")
                else:
                    raise ValueError(f"AWS authentication failed: {error_code}")
            
            return s3_client
            
        except NoCredentialsError:
            raise ValueError("AWS credentials not provided or invalid")
        except Exception as e:
            raise RuntimeError(f"Failed to create S3 client: {str(e)}")

    def _download_with_progress(self, s3_client, bucket: str, key: str, local_path: str):
        """Download file with progress tracking"""
        try:
            from botocore.exceptions import ClientError
            
            # Get object info for progress tracking
            try:
                response = s3_client.head_object(Bucket=bucket, Key=key)
                file_size = response['ContentLength']
                self._debug_log(f"📁 File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)", "INFO")
            except ClientError as e:
                if e.response['Error']['Code'] == 'NotFound':
                    raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}")
                else:
                    raise RuntimeError(f"Failed to get object info: {e.response['Error']['Code']}")
            
            # Progress tracking class
            class ProgressTracker:
                def __init__(self, total_size, logger, debug_log_func):
                    self.total_size = total_size
                    self.downloaded = 0
                    self.logger = logger
                    self.debug_log = debug_log_func
                    self.last_log_time = time.time()
                    
                def __call__(self, bytes_transferred):
                    self.downloaded += bytes_transferred
                    current_time = time.time()
                    
                    # Log progress every 2 seconds or at completion
                    if current_time - self.last_log_time >= 2 or self.downloaded == self.total_size:
                        progress = (self.downloaded / self.total_size) * 100
                        self.debug_log(f"📊 Download progress: {progress:.1f}% ({self.downloaded:,}/{self.total_size:,} bytes)", "INFO")
                        self.last_log_time = current_time
            
            # Download with progress tracking
            progress_tracker = ProgressTracker(file_size, self.logger, self._debug_log)
            
            self._debug_log(f"⬇️ Starting download from s3://{bucket}/{key}", "INFO")
            start_time = time.time()
            
            s3_client.download_file(
                bucket, 
                key, 
                local_path, 
                Callback=progress_tracker
            )
            
            download_time = time.time() - start_time
            download_speed = file_size / download_time if download_time > 0 else 0
            
            self._debug_log(f"✅ Download completed in {download_time:.2f}s (Speed: {download_speed / (1024*1024):.2f} MB/s)", "INFO")
            
            return {
                'file_size': file_size,
                'download_time': download_time,
                'download_speed': download_speed
            }
            
        except Exception as e:
            raise RuntimeError(f"Download failed: {str(e)}")

    def download_video(self) -> Data:
        """Main method to download video from S3"""
        try:
            start_time = time.time()
            
            # Check dependencies
            self._check_dependencies()
            
            # Validate S3 URL
            self._debug_log(f"🔍 Parsing S3 URL: {self.s3_url}", "INFO")
            s3_info = self._validate_s3_url(self.s3_url)
            self._debug_log(f"✅ S3 URL validation passed: {s3_info}", "INFO")
            
            # Validate video format
            if not self._validate_video_format(s3_info['filename']):
                raise ValueError(f"Unsupported video format: {Path(s3_info['filename']).suffix}")
            
            # Create local directory
            local_dir = self._create_local_directory(self.local_storage_path)
            local_file_path = Path(local_dir) / s3_info['filename']
            self._debug_log(f"✅ Local directory created: {local_dir}", "INFO")
            
            # Create S3 client
            s3_client = self._create_s3_client()
            self._debug_log("✅ S3 client created successfully", "INFO")
            
            # Download video
            download_stats = self._download_with_progress(
                s3_client,
                s3_info['bucket'],
                s3_info['key'],
                str(local_file_path)
            )
            
            # Verify download
            if not local_file_path.exists():
                raise RuntimeError("Downloaded file does not exist")
            
            actual_size = local_file_path.stat().st_size
            if actual_size != download_stats['file_size']:
                raise RuntimeError(f"File size mismatch: expected {download_stats['file_size']}, got {actual_size}")
            
            # Store results
            self.local_file_path = str(local_file_path)
            self.operation_success = True
            
            total_time = time.time() - start_time
            
            self.download_info = {
                'success': True,
                's3_url': self.s3_url,
                'bucket': s3_info['bucket'],
                'key': s3_info['key'],
                'filename': s3_info['filename'],
                'local_path': self.local_file_path,
                'local_directory': local_dir,
                'file_size_bytes': download_stats['file_size'],
                'file_size_mb': download_stats['file_size'] / (1024 * 1024),
                'download_time_seconds': download_stats['download_time'],
                'download_speed_mbps': download_stats['download_speed'] / (1024 * 1024),
                'total_processing_time_seconds': total_time,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'aws_region': self.aws_region,
                'video_format': Path(s3_info['filename']).suffix.lower()
            }
            
            self._debug_log(f"✅ Video download completed successfully: {self.local_file_path}", "INFO")
            self._debug_log(f"📊 Total processing time: {total_time:.3f}s", "INFO")
            
            return Data(data=self.download_info)
            
        except Exception as e:
            # Store failure state
            self.operation_success = False
            self.error_details = str(e)
            
            error_msg = f"Video download failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            
            error_result = {
                'success': False,
                'error': error_msg,
                's3_url': self.s3_url,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'aws_region': self.aws_region if hasattr(self, 'aws_region') else 'unknown'
            }
            
            return Data(data=error_result)
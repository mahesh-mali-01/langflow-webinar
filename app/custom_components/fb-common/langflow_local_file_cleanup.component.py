import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langflow.custom import Component
from langflow.inputs import MessageTextInput, StrInput, BoolInput, IntInput, MultilineInput, DataInput
from langflow.io import Output
from langflow.schema import Data
from langflow.schema.message import Message


class LangflowFileCleanup(Component):
    display_name = "Langflow File Cleanup"
    description = "Clean up local files and directories after processing completion"
    documentation = """
    This component safely removes local files and directories that were created during processing.
    It supports single file deletion, directory removal, batch cleanup, and provides safety checks
    to prevent accidental deletion of important files. Designed to work with the S3 Video Downloader
    component for workflow cleanup.
    """
    icon = "eraser"
    name = "LangflowFileCleanup"

    inputs = [
        MessageTextInput(
            name="file_paths",
            display_name="File Paths",
            info="File or directory paths to clean up (one per line for multiple paths)",
            required=True,
            tool_mode=True,
        ),
        DataInput(
            name="trigger_cleanup",
            display_name="Trigger Cleanup",
            info="Connect any output here to trigger cleanup (Data, Message, or any other type)",
            required=False,
            input_types=["Data", "Message", "str", "int", "bool", "float"],
        ),
        BoolInput(
            name="remove_parent_directory",
            display_name="Remove Parent Directory",
            info="Remove the parent directory if it becomes empty after cleanup",
            value=False,
        ),
        BoolInput(
            name="force_cleanup",
            display_name="Force Cleanup",
            info="Force cleanup even if files are in use (use with caution)",
            value=False,
        ),
        BoolInput(
            name="recursive_directory_removal",
            display_name="Recursive Directory Removal",
            info="Recursively remove directories and their contents",
            value=True,
        ),
        MultilineInput(
            name="protected_paths",
            display_name="Protected Paths",
            info="Additional paths that should never be deleted (one per line). System paths are protected by default.",
            value="",
            tool_mode=True,
        ),
        MultilineInput(
            name="allowed_paths",
            display_name="Allowed Paths",
            info="Paths that are explicitly allowed to be deleted, even if they might seem protected (one per line)",
            value="/app/downloads/videos",
            tool_mode=True,
        ),
        IntInput(
            name="max_file_age_hours",
            display_name="Max File Age (hours)",
            info="Only delete files older than specified hours (0 = no age limit)",
            value=0,
        ),
        BoolInput(
            name="dry_run",
            display_name="Dry Run",
            info="Preview what would be deleted without actually deleting",
            value=False,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose Logging",
            info="Enable detailed logging of cleanup operations",
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="cleanup_files")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.operation_success = False
        self.error_details = None
        self.cleaned_files = []
        self.cleanup_stats = {}
        
        # Default protected paths - only critical system directories
        # Note: We don't include '/' as it would protect everything
        self.default_protected_paths = {
            '/home', '/usr', '/var', '/etc', '/bin', '/sbin', '/lib', '/opt', '/root',
            '/proc', '/sys', '/dev', '/run', '/boot'
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

    def _parse_file_paths(self, file_paths: str) -> List[str]:
        """Parse file paths from input string"""
        try:
            # Handle different input formats
            if isinstance(file_paths, str):
                # Try to parse as JSON first
                try:
                    parsed_paths = json.loads(file_paths)
                    if isinstance(parsed_paths, list):
                        return parsed_paths
                    elif isinstance(parsed_paths, str):
                        return [parsed_paths]
                except json.JSONDecodeError:
                    pass
                
                # Split by newlines and filter empty lines
                paths = [path.strip() for path in file_paths.strip().split('\n') if path.strip()]
                return paths
            elif isinstance(file_paths, list):
                return file_paths
            else:
                return [str(file_paths)]
                
        except Exception as e:
            raise ValueError(f"Invalid file paths format: {str(e)}")

    def _parse_protected_paths(self) -> List[str]:
        """Parse protected paths from input"""
        try:
            if not self.protected_paths:
                return list(self.default_protected_paths)
            
            # Parse protected paths
            protected = []
            for path in self.protected_paths.strip().split('\n'):
                path = path.strip()
                if path:
                    protected.append(path)
            
            # Always include default protected paths
            protected.extend(self.default_protected_paths)
            
            return list(set(protected))  # Remove duplicates
            
        except Exception as e:
            self._debug_log(f"Error parsing protected paths, using defaults: {str(e)}", "WARNING")
            return list(self.default_protected_paths)

    def _parse_allowed_paths(self) -> List[str]:
        """Parse allowed paths from input"""
        try:
            if not self.allowed_paths:
                return []
            
            # Parse allowed paths
            allowed = []
            for path in self.allowed_paths.strip().split('\n'):
                path = path.strip()
                if path:
                    allowed.append(path)
            
            return allowed
            
        except Exception as e:
            self._debug_log(f"Error parsing allowed paths: {str(e)}", "WARNING")
            return []

    def _is_path_protected(self, path: str, protected_paths: List[str], allowed_paths: List[str]) -> bool:
        """Check if a path is protected from deletion"""
        try:
            self._debug_log(f"Checking protection for {path}", "INFO")
            path_obj = Path(path).resolve()
            
            # First check if path is explicitly allowed
            for allowed in allowed_paths:
                try:
                    allowed_obj = Path(allowed).resolve()
                    
                    # Check if path is exactly the allowed path or under it
                    if path_obj == allowed_obj or allowed_obj in path_obj.parents:
                        self._debug_log(f"Path is explicitly allowed: {allowed}", "INFO")
                        return False
                        
                except Exception as e:
                    self._debug_log(f"Error resolving allowed path {allowed}: {str(e)}", "WARNING")
                    continue
            
            # Then check if path is protected
            for protected in protected_paths:
                # Handle wildcard patterns
                if '*' in protected:
                    import fnmatch
                    if fnmatch.fnmatch(str(path_obj), protected):
                        self._debug_log(f"Path matches wildcard pattern: {protected}", "INFO")
                        return True
                else:
                    try:
                        protected_obj = Path(protected).resolve()
                        
                        # Check if path is exactly the protected path
                        if path_obj == protected_obj:
                            self._debug_log(f"Path matches protected path exactly: {protected}", "INFO")
                            return True
                        
                        # Check if path is under a protected directory
                        if protected_obj in path_obj.parents:
                            self._debug_log(f"Path is under protected directory: {protected}", "INFO")
                            return True
                        
                    except Exception as e:
                        self._debug_log(f"Error resolving protected path {protected}: {str(e)}", "WARNING")
                        continue
            
            self._debug_log(f"Path is not protected: {path}", "INFO")
            return False
            
        except Exception as e:
            self._debug_log(f"Error checking protection for {path}: {str(e)}", "WARNING")
            return True  # Err on the side of caution

    def _check_file_age(self, file_path: str) -> bool:
        """Check if file is older than max age"""
        try:
            if self.max_file_age_hours <= 0:
                return True  # No age limit
            
            path_obj = Path(file_path)
            if not path_obj.exists():
                return True  # File doesn't exist, can "clean"
            
            file_age = time.time() - path_obj.stat().st_mtime
            max_age_seconds = self.max_file_age_hours * 3600
            
            return file_age > max_age_seconds
            
        except Exception as e:
            self._debug_log(f"Error checking file age for {file_path}: {str(e)}", "WARNING")
            return False  # Err on the side of caution

    def _is_file_in_use(self, file_path: str) -> bool:
        """Check if file is currently in use"""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return False
            
            # Try to rename file to itself (non-destructive check)
            if path_obj.is_file():
                try:
                    path_obj.rename(path_obj)
                    return False
                except OSError:
                    return True
            
            return False
            
        except Exception as e:
            self._debug_log(f"Error checking if file in use {file_path}: {str(e)}", "WARNING")
            return True  # Err on the side of caution

    def _remove_file(self, file_path: str) -> Dict[str, Any]:
        """Remove a single file"""
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return {
                    'path': file_path,
                    'status': 'skipped',
                    'reason': 'File does not exist',
                    'size_bytes': 0
                }
            
            # Get file size before deletion
            file_size = path_obj.stat().st_size if path_obj.is_file() else 0
            
            if self.dry_run:
                return {
                    'path': file_path,
                    'status': 'would_delete',
                    'reason': 'Dry run mode',
                    'size_bytes': file_size
                }
            
            # Remove file
            if path_obj.is_file():
                path_obj.unlink()
                self._debug_log(f"🗑️ Removed file: {file_path}", "INFO")
            elif path_obj.is_dir():
                if self.recursive_directory_removal:
                    shutil.rmtree(path_obj)
                    self._debug_log(f"🗑️ Removed directory: {file_path}", "INFO")
                else:
                    path_obj.rmdir()
                    self._debug_log(f"🗑️ Removed empty directory: {file_path}", "INFO")
            
            return {
                'path': file_path,
                'status': 'deleted',
                'reason': 'Successfully removed',
                'size_bytes': file_size
            }
            
        except PermissionError:
            return {
                'path': file_path,
                'status': 'failed',
                'reason': 'Permission denied',
                'size_bytes': 0
            }
        except OSError as e:
            return {
                'path': file_path,
                'status': 'failed',
                'reason': f'OS error: {str(e)}',
                'size_bytes': 0
            }
        except Exception as e:
            return {
                'path': file_path,
                'status': 'failed',
                'reason': f'Unexpected error: {str(e)}',
                'size_bytes': 0
            }

    def _remove_empty_parent_directory(self, file_path: str, protected_paths: List[str], allowed_paths: List[str]) -> Optional[Dict[str, Any]]:
        """Remove parent directory if it becomes empty"""
        try:
            if not self.remove_parent_directory:
                return None
            
            parent_dir = Path(file_path).parent
            
            # Check if parent directory is empty
            if parent_dir.exists() and parent_dir.is_dir():
                if not any(parent_dir.iterdir()):  # Directory is empty
                    if not self._is_path_protected(str(parent_dir), protected_paths, allowed_paths):
                        if self.dry_run:
                            return {
                                'path': str(parent_dir),
                                'status': 'would_delete',
                                'reason': 'Empty parent directory (dry run)',
                                'size_bytes': 0
                            }
                        else:
                            parent_dir.rmdir()
                            self._debug_log(f"🗑️ Removed empty parent directory: {parent_dir}", "INFO")
                            return {
                                'path': str(parent_dir),
                                'status': 'deleted',
                                'reason': 'Empty parent directory removed',
                                'size_bytes': 0
                            }
            
            return None
            
        except Exception as e:
            self._debug_log(f"Error removing parent directory: {str(e)}", "WARNING")
            return None

    def cleanup_files(self) -> Data:
        """Main method to clean up files"""
        try:
            # Check if cleanup is triggered - accept any non-None value
            if self.trigger_cleanup is None:
                return Data(data={'success': False, 'error': 'Cleanup not triggered - no input provided', 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")})

            start_time = time.time()
            
            # Parse input paths
            file_paths = self._parse_file_paths(self.file_paths)
            self._debug_log(f"Parsed file paths: {file_paths}", "INFO")
            protected_paths = self._parse_protected_paths()
            self._debug_log(f"Parsed protected paths: {protected_paths}", "INFO")
            allowed_paths = self._parse_allowed_paths()
            self._debug_log(f"Parsed allowed paths: {allowed_paths}", "INFO")
            
            self._debug_log(f"🧹 Starting cleanup of {len(file_paths)} path(s)", "INFO")
            
            if self.dry_run:
                self._debug_log("🔍 Running in DRY RUN mode - no files will be deleted", "INFO")
            
            cleanup_results = []
            total_size_deleted = 0
            files_deleted = 0
            files_skipped = 0
            files_failed = 0
            
            # Process each file path
            for file_path in file_paths:
                try:
                    # Resolve path
                    resolved_path = str(Path(file_path).resolve())
                    
                    # Safety checks
                    if self._is_path_protected(resolved_path, protected_paths, allowed_paths):
                        result = {
                            'path': file_path,
                            'status': 'skipped',
                            'reason': 'Path is protected',
                            'size_bytes': 0
                        }
                        cleanup_results.append(result)
                        files_skipped += 1
                        self._debug_log(f"⚠️ Skipped protected path: {file_path}", "WARNING")
                        continue
                    
                    # Check file age
                    if not self._check_file_age(resolved_path):
                        result = {
                            'path': file_path,
                            'status': 'skipped',
                            'reason': f'File is newer than {self.max_file_age_hours} hours',
                            'size_bytes': 0
                        }
                        cleanup_results.append(result)
                        files_skipped += 1
                        self._debug_log(f"⚠️ Skipped file (too new): {file_path}", "WARNING")
                        continue
                    
                    # Check if file is in use
                    if not self.force_cleanup and self._is_file_in_use(resolved_path):
                        result = {
                            'path': file_path,
                            'status': 'skipped',
                            'reason': 'File is currently in use',
                            'size_bytes': 0
                        }
                        cleanup_results.append(result)
                        files_skipped += 1
                        self._debug_log(f"⚠️ Skipped file (in use): {file_path}", "WARNING")
                        continue
                    
                    # Remove file
                    result = self._remove_file(resolved_path)
                    cleanup_results.append(result)
                    
                    # Update statistics
                    if result['status'] == 'deleted' or result['status'] == 'would_delete':
                        files_deleted += 1
                        total_size_deleted += result['size_bytes']
                    elif result['status'] == 'skipped':
                        files_skipped += 1
                    else:
                        files_failed += 1
                    
                    # Try to remove empty parent directory
                    parent_result = self._remove_empty_parent_directory(resolved_path, protected_paths, allowed_paths)
                    if parent_result:
                        cleanup_results.append(parent_result)
                        if parent_result['status'] == 'deleted' or parent_result['status'] == 'would_delete':
                            files_deleted += 1
                    
                except Exception as e:
                    result = {
                        'path': file_path,
                        'status': 'failed',
                        'reason': f'Error processing path: {str(e)}',
                        'size_bytes': 0
                    }
                    cleanup_results.append(result)
                    files_failed += 1
                    self._debug_log(f"❌ Error processing {file_path}: {str(e)}", "ERROR")
            
            # Store results
            self.cleaned_files = cleanup_results
            self.operation_success = files_failed == 0
            
            total_time = time.time() - start_time
            
            self.cleanup_stats = {
                'success': self.operation_success,
                'dry_run': self.dry_run,
                'total_paths_processed': len(file_paths),
                'files_deleted': files_deleted,
                'files_skipped': files_skipped,
                'files_failed': files_failed,
                'total_size_deleted_bytes': total_size_deleted,
                'total_size_deleted_mb': total_size_deleted / (1024 * 1024),
                'processing_time_seconds': total_time,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'protected_paths_count': len(protected_paths),
                'cleanup_results': cleanup_results
            }
            
            status = "DRY RUN" if self.dry_run else "COMPLETED"
            self._debug_log(f"✅ Cleanup {status}: {files_deleted} deleted, {files_skipped} skipped, {files_failed} failed", "INFO")
            self._debug_log(f"📊 Total size freed: {total_size_deleted / (1024*1024):.2f} MB", "INFO")
            self._debug_log(f"📊 Processing time: {total_time:.3f}s", "INFO")
            
            return Data(data=self.cleanup_stats)
            
        except Exception as e:
            # Store failure state
            self.operation_success = False
            self.error_details = str(e)
            
            error_msg = f"Cleanup failed: {str(e)}"
            self._debug_log(error_msg, "ERROR")
            
            error_result = {
                'success': False,
                'error': error_msg,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'dry_run': self.dry_run if hasattr(self, 'dry_run') else False
            }
            
            return Data(data=error_result)

    def get_cleaned_files(self) -> Message:
        """Get the list of cleaned files"""
        if self.cleaned_files:
            return Message(text=json.dumps(self.cleaned_files, indent=2))
        else:
            return Message(text="[]")

    def get_success(self) -> Message:
        """Get the success status of the cleanup operation"""
        return Message(text=str(self.operation_success))

    def get_error_details(self) -> Message:
        """Get error details if the operation failed"""
        if self.error_details:
            return Message(text=self.error_details)
        else:
            return Message(text="")
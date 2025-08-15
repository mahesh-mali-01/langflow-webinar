import json
import re
import socket
import subprocess
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import httpx
from langflow.custom import Component
from langflow.inputs import MessageTextInput, DropdownInput, IntInput, BoolInput, MultilineInput, DataInput
from langflow.schema import Data
from langflow.template import Output


class SmartAPIRequestComponent(Component):
    display_name = "Smart API Request"
    description = "API Request with automatic Docker service name to IP resolution"
    icon = "webhook"

    inputs = [
        MessageTextInput(
            name="url",
            display_name="URL",
            info="URL with service name (will auto-resolve to IP)",
            value="http://api-asset-inspection:8000/api/v1/health",
            required=True
        ),
        DropdownInput(
            name="method",
            display_name="HTTP Method",
            info="The HTTP method to use",
            options=["GET", "POST", "PUT", "PATCH", "DELETE"],
            value="POST"
        ),
        DataInput(
            name="request_data",
            display_name="Request Data",
            info="Data object from previous component (will be converted to JSON)",
            required=False
        ),
        MessageTextInput(
            name="authorization",
            display_name="Authorization Header",
            info="Authorization header (e.g., Bearer token)",
            value=""
        ),
        MessageTextInput(
            name="x_api_key",
            display_name="X-API-key Header",
            info="X-API-key header",
            value=""
        ),
        BoolInput(
            name="auto_resolve_ip",
            display_name="Auto Resolve Service Names",
            info="Automatically resolve Docker service names to IPs",
            value=True
        ),
        IntInput(
            name="timeout",
            display_name="Timeout (seconds)",
            info="Request timeout in seconds",
            value=30
        )
    ]

    outputs = [
        Output(display_name="API Response", name="api_response", method="build"),
    ]

    def build(self) -> Data:
        """Make the API request with smart IP resolution"""
        try:
            original_url = self.url
            resolved_url = self._resolve_url() if self.auto_resolve_ip else self.url

            print(f"🌐 Original URL: {original_url}")
            if resolved_url != original_url:
                print(f"🔍 Resolved URL: {resolved_url}")

            # Prepare headers
            headers = self._prepare_headers()

            # Prepare request body
            request_data = self._prepare_request_body()

            # Make the HTTP request
            response_data = self._make_http_request(resolved_url, headers, request_data)

            print(f"✅ Request completed successfully")
            return Data(
                data=response_data,
                text=json.dumps(response_data, indent=2)
            )

        except Exception as e:
            error_msg = f"API Request failed: {str(e)}"
            print(f"❌ ERROR: {error_msg}")

            error_response = {
                "success": False,
                "error": error_msg,
                "original_url": self.url,
                "method": self.method,
                "status_code": None,
                "response": None
            }

            return Data(
                data=error_response,
                text=f"API Request Error: {error_msg}"
            )

    def _resolve_url(self) -> str:
        """Resolve Docker service names to IP addresses"""
        try:
            parsed = urlparse(self.url)
            hostname = parsed.hostname

            if not hostname:
                return self.url

            # Skip resolution for localhost and IP addresses
            if hostname == 'localhost' or self._is_ip_address(hostname):
                print(f"🔍 Skipping resolution for {hostname} (localhost/IP)")
                return self.url

            # Try multiple resolution methods
            resolved_ip = (
                    self._resolve_via_socket(hostname) or
                    self._resolve_via_docker_inspect(hostname) or
                    self._resolve_via_ping(hostname)
            )

            if resolved_ip:
                # Replace hostname with IP in URL
                new_netloc = f"{resolved_ip}:{parsed.port}" if parsed.port else resolved_ip
                resolved_url = parsed._replace(netloc=new_netloc).geturl()
                print(f"🎯 Resolved {hostname} → {resolved_ip}")
                return resolved_url
            else:
                print(f"⚠️ Could not resolve {hostname}, using original URL")
                return self.url

        except Exception as e:
            print(f"⚠️ Resolution failed: {e}, using original URL")
            return self.url

    def _is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is already an IP address"""
        try:
            socket.inet_aton(hostname)
            return True
        except socket.error:
            return False

    def _resolve_via_socket(self, hostname: str) -> Optional[str]:
        """Try to resolve using Python's socket library"""
        try:
            ip = socket.gethostbyname(hostname)
            print(f"🔍 Socket resolution: {hostname} → {ip}")
            return ip
        except socket.gaierror:
            print(f"🔍 Socket resolution failed for {hostname}")
            return None

    def _resolve_via_docker_inspect(self, hostname: str) -> Optional[str]:
        """Try to resolve using docker inspect command"""
        try:
            # Try to get IP from docker inspect
            result = subprocess.run(
                ['docker', 'inspect', hostname, '--format', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                ip = result.stdout.strip()
                print(f"🔍 Docker inspect resolution: {hostname} → {ip}")
                return ip
            else:
                print(f"🔍 Docker inspect failed for {hostname}")
                return None

        except Exception as e:
            print(f"🔍 Docker inspect error: {e}")
            return None

    def _resolve_via_ping(self, hostname: str) -> Optional[str]:
        """Try to resolve using ping command"""
        try:
            # Use ping to resolve hostname
            result = subprocess.run(
                ['ping', '-c', '1', hostname],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Extract IP from ping output
                match = re.search(r'PING .* \((\d+\.\d+\.\d+\.\d+)\)', result.stdout)
                if match:
                    ip = match.group(1)
                    print(f"🔍 Ping resolution: {hostname} → {ip}")
                    return ip

            print(f"🔍 Ping resolution failed for {hostname}")
            return None

        except Exception as e:
            print(f"🔍 Ping error: {e}")
            return None

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Langflow-Smart-API-Request/1.0"
        }

        # Add Authorization if specified
        if self.authorization and self.authorization.strip():
            headers["Authorization"] = self.authorization.strip()

        if self.x_api_key and self.x_api_key.strip():
            headers["x-api-key"] = self.x_api_key.strip()

        return headers

    def _prepare_request_body(self) -> Any:
        """Prepare request body for POST/PUT/PATCH requests"""
        if self.method.upper() in ["GET", "DELETE"]:
            return None

        if not self.request_data or not hasattr(self.request_data, 'data'):
            return None

        try:
            request_body = self.request_data.data
            return request_body
        except json.JSONDecodeError:
            print("⚠️ Request body is not valid JSON")
            return self.request_body

    def _make_http_request(self, url: str, headers: Dict[str, str], request_data: Any) -> Dict[str, Any]:
        """Make the actual HTTP request"""

        try:
            with httpx.Client(timeout=self.timeout) as client:
                # Make request based on method
                if self.method.upper() == "GET":
                    response = client.get(url, headers=headers)
                elif self.method.upper() == "POST":
                    response = client.post(url, headers=headers, json=request_data)
                elif self.method.upper() == "PUT":
                    response = client.put(url, headers=headers, json=request_data)
                elif self.method.upper() == "PATCH":
                    response = client.patch(url, headers=headers, json=request_data)
                elif self.method.upper() == "DELETE":
                    response = client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.method}")

                print(f"📡 Response Status: {response.status_code}")

                # Parse response
                try:
                    response_content = response.json()
                except json.JSONDecodeError:
                    response_content = response.text

                # Build result
                result = {
                    "success": True,
                    "url": url,
                    "original_url": self.url,
                    "method": self.method.upper(),
                    "status_code": response.status_code,
                    "response": response_content,
                    "response_headers": dict(response.headers),
                    "elapsed_time": response.elapsed.total_seconds()
                }

                return result

        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

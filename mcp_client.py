"""MCP Client for connecting to Model Context Protocol servers"""

import requests
import json
import uuid
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Generator, Any, Tuple, List
import sseclient
import re
from config import get_mcp_server_url, get_health_check_url, MCP_SERVER_CONFIG
from ai_processor import AIProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClientError(Exception):
    """Custom exception for MCP client errors"""
    pass

class MCPClient:
    """Client for interacting with MCP (Model Context Protocol) servers"""
    
    def __init__(self, server_url: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize MCP client
        
        Args:
            server_url: Optional custom server URL, defaults to config value
            openai_api_key: Optional OpenAI API key for AI processing
        """
        self.base_url = MCP_SERVER_CONFIG["base_url"]
        self.sse_url = get_mcp_server_url()
        self.health_url = get_health_check_url()
        self.messages_url = f"{self.base_url}{MCP_SERVER_CONFIG['messages_endpoint']}"
        self.timeout = MCP_SERVER_CONFIG["timeout"]
        self.max_retries = MCP_SERVER_CONFIG["max_retries"]
        self.session_id = None  # Will be obtained from SSE endpoint
        
        # Initialize AI processor if API key provided
        self.ai_processor = None
        if openai_api_key:
            try:
                self.ai_processor = AIProcessor(openai_api_key)
                logger.info("AI processor initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize AI processor: {e}")
        
        # Store last response for context
        self.last_response = None
        
        # Set up session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        })
        
        # Define available tools and their patterns
        # Let's try common variations of tool names that might exist in your n8n setup
        self.tool_patterns = {
            r'(?i)\b(list\s+do|list\s+delivery\s+orders?|show\s+do|show\s+delivery\s+orders?)\b': {
                'tool': 'list_do',  # Try simpler name first
                'alternatives': ['list_delivery_orders', 'listDO', 'list-do', 'get_delivery_orders'],
                'description': 'Returns a list of all existing Delivery Orders (DOs) with status, ID, and dates.'
            },
            r'(?i)\b(change\s+.*status.*DO-\w+.*(?:approved|pending|rejected|completed))\b': {
                'tool': 'update_status',
                'alternatives': ['update_delivery_order_status', 'updateStatus', 'change_status'],
                'description': 'Updates status of {DO_number} to specified status'
            },
            r'(?i)\b(generate\s+do|create\s+do|generate\s+delivery\s+order|create\s+delivery\s+order)\b': {
                'tool': 'generate_do',
                'alternatives': ['generate_delivery_order', 'generateDO', 'create_do'],
                'description': 'Triggers generation of corresponding DO PDF'
            },
            r'(?i)\b(check\s+holding\s+area|list\s+holding\s+area|show\s+holding\s+area|holding\s+area)\b': {
                'tool': 'holding_area',
                'alternatives': ['check_holding_area', 'list_holding_area', 'holdingArea', 'get_holding_area'],
                'description': 'Displays all items or orders currently in the holding area queue'
            },
            r'(?i)\b(generate\s+invoice|create\s+invoice)\b': {
                'tool': 'generate_invoice',
                'alternatives': ['generateInvoice', 'create_invoice'],
                'description': 'Triggers generation of corresponding Invoice PDF'
            },
            r'(?i)\b(generate\s+po|create\s+po|generate\s+purchase\s+order|create\s+purchase\s+order)\b': {
                'tool': 'generate_po',
                'alternatives': ['generate_purchase_order', 'generatePO', 'create_po'],
                'description': 'Triggers generation of corresponding PO PDF'
            }
        }
        
        logger.info(f"MCP Client initialized with base URL: {self.base_url}")
        logger.info(f"SSE URL: {self.sse_url}")
        logger.info(f"Messages URL: {self.messages_url}")
    
    def _map_query_to_tool(self, message: str) -> Tuple[str, Dict]:
        """
        Map user query to appropriate MCP tool
        
        Args:
            message: User's query
            
        Returns:
            Tuple of (tool_name, arguments)
        """
        message_lower = message.lower().strip()
        
        for pattern, tool_info in self.tool_patterns.items():
            if re.search(pattern, message):
                tool_name = tool_info['tool']
                arguments = {}
                
                # Extract specific parameters based on the tool
                if tool_name == 'update_delivery_order_status':
                    # Extract DO number and status
                    do_match = re.search(r'DO-(\w+)', message, re.IGNORECASE)
                    status_match = re.search(r'\b(approved|pending|rejected|completed)\b', message, re.IGNORECASE)
                    
                    if do_match:
                        arguments['do_number'] = f"DO-{do_match.group(1)}"
                    if status_match:
                        arguments['status'] = status_match.group(1).upper()
                
                elif tool_name in ['generate_delivery_order', 'generate_invoice', 'generate_purchase_order']:
                    # These might need additional parameters - can be extracted from message
                    # For now, pass the original message for context
                    arguments['context'] = message
                
                logger.info(f"Mapped query to tool: {tool_name} with arguments: {arguments}")
                return tool_name, arguments
        
        # If no specific tool is matched, return a default or error
        logger.warning(f"No tool mapping found for query: {message}")
        return "unknown_tool", {"message": message}
    
    def _get_tool_alternatives(self, tool_name: str) -> List[str]:
        """
        Get alternative tool names for a given tool
        
        Args:
            tool_name: The primary tool name
            
        Returns:
            List of alternative tool names
        """
        for pattern, tool_info in self.tool_patterns.items():
            if tool_info['tool'] == tool_name:
                return tool_info.get('alternatives', [])
        return []
    
    def _should_handle_with_ai_only(self, message: str) -> bool:
        """
        Check if query should be handled by AI directly without going to MCP server
        
        Args:
            message: User's query
            
        Returns:
            True if AI should handle this directly
        """
        conversational_keywords = [
            'table format', 'tabular', 'table', 'format', 'columns', 'rows',
            'show me', 'give me', 'can you', 'display', 'present',
            'overdue', 'analyze', 'analysis', 'summary', 'summarize',
            'count', 'total', 'group by', 'filter', 'sort',
            'above result', 'previous result', 'that result', 'from above',
            'the data', 'this data', 'those items', 'these items'
        ]
        
        message_lower = message.lower()
        
        # If message contains conversational keywords AND we have previous data
        has_conversational_keywords = any(keyword in message_lower for keyword in conversational_keywords)
        has_previous_data = self.last_response is not None
        
        # Specifically handle "table format" requests
        is_table_request = any(keyword in message_lower for keyword in ['table', 'format', 'tabular'])
        
        return (has_conversational_keywords and has_previous_data) or is_table_request
    
    def _map_query_to_specific_tool(self, message: str) -> str:
        """
        Map user queries to specific tool names discovered from the server
        
        Args:
            message: User query
            
        Returns:
            Tool name to call
        """
        message_lower = message.lower().strip()
        
        # Direct mapping to server tools (from tools/list response)
        if any(keyword in message_lower for keyword in ['holding area', 'holding']):
            return 'Check_Holding_Area'
        elif any(keyword in message_lower for keyword in ['check do', 'list do', 'delivery order']):
            return 'Check_DO'
        elif any(keyword in message_lower for keyword in ['treatment', 'full treatment']):
            return 'Check_Full_Treatment'
        elif any(keyword in message_lower for keyword in ['ncr', 'non-conformance', 'non conformance', 'check ncr', 'ncr queries', 'ncr list']):
            return 'Check_full_ncr'
        else:
            # Default to Agent tool for complex queries
            return 'Agent'
    
    def get_available_commands(self) -> Dict[str, str]:
        """
        Get a dictionary of available commands based on actual server tools
        
        Returns:
            Dict mapping command examples to descriptions
        """
        commands = {
            "Check Holding Area": "Get rows from Holding Area sheet in Google Sheets",
            "Check DO": "Get rows from Delivery Order sheet in Google Sheets", 
            "Check Full Treatment": "Get rows from Full Treatment sheet in Google Sheets",
            "Check Full NCR": "Get NCR (Non-Conformance Report) list (direct tool)",
            "General Query": "Ask complex questions (routed through AI Agent)"
        }
        return commands
    
    def _get_session_id(self) -> Optional[str]:
        """
        Get session ID from the SSE endpoint
        
        Returns:
            Session ID if available, None otherwise
        """
        try:
            response = self.session.get(self.sse_url, stream=True, timeout=10)
            if response.status_code == 200:
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.event == "endpoint" and event.data:
                        # Parse sessionId from data like: /mcp/ClaudeDesktopMCP/messages?sessionId=46fe7f7e-cfe6-42b7-b48e-7e05d64294a4
                        if "sessionId=" in event.data:
                            session_id = event.data.split("sessionId=")[1]
                            logger.info(f"Obtained session ID: {session_id}")
                            return session_id
                        break  # Stop after first event
            return None
        except Exception as e:
            logger.error(f"Failed to get session ID: {str(e)}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the MCP server is healthy by testing the SSE endpoint
        
        Returns:
            Dict containing health status information
        """
        try:
            # Test the SSE endpoint directly since that's what works
            start_time = time.time()
            response = self.session.get(self.sse_url, stream=True, timeout=10)
            
            if response.status_code == 200:
                # Try to read the first event to confirm it's working
                try:
                    client = sseclient.SSEClient(response)
                    for event in client.events():
                        if event.event == "endpoint":
                            response_time = time.time() - start_time
                            return {
                                "status": "healthy",
                                "message": "SSE endpoint responding",
                                "response_time": response_time
                            }
                        break  # Just check first event
                    
                    # If we get here, SSE is working but no endpoint event
                    response_time = time.time() - start_time
                    return {
                        "status": "healthy",
                        "message": "SSE endpoint working",
                        "response_time": response_time
                    }
                except Exception as sse_error:
                    response_time = time.time() - start_time
                    return {
                        "status": "unhealthy",
                        "message": f"SSE parsing error: {str(sse_error)}",
                        "response_time": response_time
                    }
            else:
                response_time = time.time() - start_time
                return {
                    "status": "unhealthy",
                    "message": f"SSE endpoint returned status {response.status_code}",
                    "response_time": response_time
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "message": "SSE endpoint timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "connection_error",
                "message": "Could not connect to SSE endpoint"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }
    
    def send_query(self, message: str, conversation_id: Optional[str] = None, 
                   stream: bool = True, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Send a query to the MCP server with intelligent routing
        
        Steps:
        1. Check if AI should handle this query
        2. Either process with AI or send to MCP server
        3. Store response for context
        
        Args:
            message: The user's message/query
            conversation_id: Optional conversation ID for context
            stream: Whether to use streaming response
            conversation_context: Recent conversation for AI context
            
        Returns:
            Dict containing the response or error information
        """
        if not message.strip():
            return {
                "success": False,
                "error": "Message cannot be empty"
            }
        
        # Check if this is a conversational query that should be handled by AI directly
        # (like "table format", "show overdue", "analyze", etc.)
        if self.ai_processor and self._should_handle_with_ai_only(message):
            logger.info(f"Routing conversational query directly to AI: {message[:100]}...")
            
            ai_result = self.ai_processor.process_query(
                message,
                self.last_response,
                conversation_context or []
            )
            
            return ai_result
        
        # First, get the raw data from MCP server
        conversation_id = conversation_id or str(uuid.uuid4())
        logger.info(f"Sending query to MCP server: {message[:100]}...")
        
        # Get raw data from MCP
        raw_result = None
        for attempt in range(self.max_retries):
            try:
                raw_result = self._send_query_with_sse(message.strip(), conversation_id, attempt + 1)
                if raw_result.get("success"):
                    break
            except Exception as e:
                logger.warning(f"MCP query attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": f"Failed to get data from MCP server: {str(e)}"
                    }
        
        # If we have AI processor, let it handle ALL formatting (like Claude Desktop)
        if self.ai_processor and raw_result and raw_result.get("success"):
            logger.info(f"Processing response with OpenAI for user-friendly formatting...")
            
            try:
                # Let OpenAI format the response into human-readable text
                ai_result = self.ai_processor.process_mcp_response(
                    message,
                    raw_result,
                    conversation_context or []
                )
                
                # Store the raw result for future reference
                self.last_response = raw_result
                
                return ai_result
                
            except Exception as e:
                logger.error(f"AI formatting failed: {e}")
                # Fallback to raw result if AI fails
                return raw_result
        
        # If no AI processor, return raw result
        if raw_result:
            self.last_response = raw_result
            return raw_result
        
        # Fallback if no raw result
        return {
            "success": False,
            "error": "Failed to get response from MCP server"
        }
    
    def _send_query_with_sse(self, message: str, conversation_id: str, attempt: int) -> Dict[str, Any]:
        """
        Send query using proper SSE protocol:
        1. Establish SSE connection to get session ID
        2. Send message to messages endpoint
        3. Parse response from SSE stream
        """
        try:
            # Step 1: Establish SSE connection and get session ID
            logger.info("Establishing SSE connection...")
            sse_response = self.session.get(self.sse_url, stream=True, timeout=30)
            
            if sse_response.status_code != 200:
                error_text = sse_response.text[:500] if sse_response.text else "No error message"
                return {
                    "success": False,
                    "error": f"Failed to establish SSE connection: HTTP {sse_response.status_code}: {error_text}",
                    "attempt": attempt
                }
            
            # Step 2: Parse SSE events to get session ID and send message
            client = sseclient.SSEClient(sse_response)
            session_id = None
            full_response = ""
            start_time = time.time()
            
            for event in client.events():
                if event.event == "endpoint" and event.data:
                    # Extract session ID from endpoint data
                    if "sessionId=" in event.data:
                        session_id = event.data.split("sessionId=")[1]
                        logger.info(f"Got session ID: {session_id}")
                        
                        # Step 3: Send message to messages endpoint
                        messages_url = f"{self.messages_url}?sessionId={session_id}"
                        
                        # Based on tools discovery, we can call specific tools directly
                        # Map queries to the correct tool names from the server
                        tool_name = self._map_query_to_specific_tool(message)
                        
                        # Set arguments based on tool type
                        if tool_name == 'Agent':
                            # Agent tool expects "input" parameter (not "query")
                            arguments = {"input": message}
                        else:
                            # Google Sheets tools don't need arguments
                            arguments = {}
                        
                        payload = {
                            "jsonrpc": "2.0",
                            "id": str(uuid.uuid4()),
                            "method": "tools/call",
                            "params": {
                                "name": tool_name,
                                "arguments": arguments
                            }
                        }
                        
                        logger.info(f"Sending message to: {messages_url}")
                        
                        # Send message in a separate request (non-blocking)
                        import threading
                        def send_message():
                            try:
                                msg_response = self.session.post(messages_url, json=payload, timeout=10)
                                logger.info(f"Message sent, status: {msg_response.status_code}")
                                if msg_response.status_code != 200:
                                    logger.error(f"Message error: {msg_response.text}")
                            except Exception as e:
                                logger.error(f"Failed to send message: {e}")
                        
                        threading.Thread(target=send_message, daemon=True).start()
                        
                elif event.data and event.data.strip():
                    # Collect response data
                    try:
                        data = json.loads(event.data)
                        if isinstance(data, dict):
                            # Store the complete structured response for AI context
                            if 'result' in data:
                                return {
                                    "success": True,
                                    "response": data.get('result', {}),
                                    "result": data.get('result', {}),  # Store for AI processing
                                    "timestamp": datetime.now().isoformat(),
                                    "attempt": attempt
                                }
                            elif 'content' in data:
                                full_response += str(data['content'])
                            elif 'message' in data:
                                full_response += str(data['message'])
                            elif 'text' in data:
                                full_response += str(data['text'])
                            elif 'response' in data:
                                full_response += str(data['response'])
                            else:
                                full_response += str(data)
                        else:
                            full_response += str(data)
                    except json.JSONDecodeError:
                        full_response += event.data
                    
                    # If we have a response and session was established, we can return
                    if session_id and full_response.strip():
                        break
                
                # Timeout check - don't wait forever
                if time.time() - start_time > 25:  # 25 second timeout
                    break
            
            if not session_id:
                return {
                    "success": False,
                    "error": "Could not establish session with server",
                    "attempt": attempt
                }
            
            if full_response.strip():
                return {
                    "success": True,
                    "response": full_response,
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt
                }
            else:
                return {
                    "success": True,
                    "response": "Message sent successfully (no response received yet)",
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt
                }
                
        except requests.exceptions.RequestException as e:
            raise e
    
    def _send_streaming_query(self, payload: Dict, attempt: int) -> Dict[str, Any]:
        """Send query with streaming response"""
        try:
            response = self.session.post(
                self.server_url,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return self._parse_sse_response(response)
            else:
                error_text = response.text[:500] if response.text else "No error message"
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "attempt": attempt
                }
                
        except requests.exceptions.RequestException as e:
            raise e
    
    def _send_regular_query(self, payload: Dict, attempt: int) -> Dict[str, Any]:
        """Send query with regular response"""
        try:
            # Remove stream parameter for regular requests
            regular_payload = {k: v for k, v in payload.items() if k != "stream"}
            
            response = self.session.post(
                self.server_url.replace("/sse", ""),
                json=regular_payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return {
                        "success": True,
                        "response": data.get("response", str(data)),
                        "timestamp": datetime.now().isoformat(),
                        "attempt": attempt
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "response": response.text,
                        "timestamp": datetime.now().isoformat(),
                        "attempt": attempt
                    }
            else:
                error_text = response.text[:500] if response.text else "No error message"
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "attempt": attempt
                }
                
        except requests.exceptions.RequestException as e:
            raise e
    
    def _parse_sse_response(self, response) -> Dict[str, Any]:
        """
        Parse Server-Sent Events response
        
        Args:
            response: The streaming response object
            
        Returns:
            Dict containing parsed response data
        """
        try:
            client = sseclient.SSEClient(response)
            full_response = ""
            metadata = {}
            
            for event in client.events():
                if event.data and event.data.strip():
                    try:
                        # Try to parse as JSON
                        data = json.loads(event.data)
                        
                        # Extract content from various possible fields
                        if isinstance(data, dict):
                            if 'content' in data:
                                full_response += str(data['content'])
                            elif 'message' in data:
                                full_response += str(data['message'])
                            elif 'text' in data:
                                full_response += str(data['text'])
                            elif 'response' in data:
                                full_response += str(data['response'])
                            elif 'data' in data:
                                full_response += str(data['data'])
                            else:
                                # If no recognized content field, use the whole object
                                full_response += str(data)
                            
                            # Store metadata
                            if 'type' in data:
                                metadata['type'] = data['type']
                            if 'status' in data:
                                metadata['status'] = data['status']
                                
                        else:
                            # If not a dict, just append the data
                            full_response += str(data)
                            
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        full_response += event.data
            
            if not full_response.strip():
                return {
                    "success": False,
                    "error": "Received empty response from server"
                }
            
            # Auto-format JSON responses for better readability
            formatted_response = self._format_response_for_display(full_response, metadata)
            
            return {
                "success": True,
                "response": formatted_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to parse SSE response: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to parse server response: {str(e)}"
            }
    
    def _format_response_for_display(self, response: str, metadata: Dict) -> str:
        """
        Format response for better user readability
        
        Args:
            response: Raw response string
            metadata: Response metadata
            
        Returns:
            Formatted response string
        """
        try:
            logger.info(f"Formatting response: {response[:200]}...")
            
            # First, try to extract JSON data from the nested structure
            extracted_data = self._extract_json_data_from_response(response)
            
            if extracted_data:
                logger.info(f"Extracted {len(extracted_data)} items from response")
                
                # Check if it's holding area data
                if all('POnumber' in item and 'productCode' in item for item in extracted_data[:3]):
                    return self._format_holding_area_data(extracted_data)
                else:
                    return self._format_generic_json_data(extracted_data)
            
            # If no data extracted, return original response
            logger.warning("No JSON data extracted, returning original response")
            return response
            
        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            return response
    
    def _extract_json_from_last_response(self) -> Optional[List[Dict]]:
        """
        Extract JSON data from the last stored response
        
        Returns:
            List of dictionaries if found, None otherwise
        """
        if not self.last_response:
            return None
            
        try:
            # Try different response structures
            response_str = str(self.last_response.get('response', ''))
            
            # Use the existing extraction method
            return self._extract_json_data_from_response(response_str)
            
        except Exception as e:
            logger.error(f"Error extracting JSON from last response: {e}")
            return None
    
    def _extract_json_data_from_response(self, response: str) -> Optional[List[Dict]]:
        """
        Extract JSON array data from nested response structure
        
        Args:
            response: Raw response string
            
        Returns:
            List of dictionaries if found, None otherwise
        """
        try:
            import re
            
            # Pattern 1: Look for nested JSON in content structure
            # {"content":[{"type":"text","text":"[{...}]"}]}
            content_pattern = r'"text":"(\[{.*?}\])"'
            content_match = re.search(content_pattern, response, re.DOTALL)
            
            if content_match:
                json_str = content_match.group(1)
                # Unescape the JSON string
                json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
                logger.info(f"Found nested JSON pattern, length: {len(json_str)}")
                
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and len(data) > 0:
                        return data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse extracted JSON: {e}")
            
            # Pattern 2: Direct JSON array pattern
            direct_pattern = r'\[{.*?}\]'
            direct_match = re.search(direct_pattern, response, re.DOTALL)
            
            if direct_match:
                json_str = direct_match.group(0)
                logger.info(f"Found direct JSON pattern, length: {len(json_str)}")
                
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and len(data) > 0:
                        return data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse direct JSON: {e}")
            
            # Pattern 3: Try to parse the entire response as JSON first
            try:
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict):
                    # Look for content array
                    if 'content' in parsed_response:
                        content = parsed_response['content']
                        if isinstance(content, list) and len(content) > 0:
                            for item in content:
                                if 'text' in item:
                                    text_content = item['text']
                                    if text_content.startswith('[{'):
                                        try:
                                            data = json.loads(text_content)
                                            if isinstance(data, list) and len(data) > 0:
                                                return data
                                        except:
                                            continue
            except json.JSONDecodeError:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return None
    
    def _format_holding_area_data(self, data: list) -> str:
        """Format holding area data for display"""
        try:
            # Create a summary view
            summary = f"## 📦 Holding Area Summary\n\n**Total Items:** {len(data)}\n\n"
            
            # Group by status
            status_counts = {}
            for item in data:
                status = item.get('Stock Status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            summary += "**Stock Status Breakdown:**\n"
            for status, count in status_counts.items():
                emoji = "❌" if status == "Not Found" else "⚠️" if status == "Insufficient" else "✅"
                summary += f"- {emoji} {status}: {count} items\n"
            
            summary += "\n**Recent Items (Last 5):**\n\n"
            
            # Show last 5 items in a readable format
            for i, item in enumerate(data[-5:], 1):
                po_num = item.get('POnumber', 'N/A')
                product = item.get('productCode', 'N/A')
                status = item.get('Stock Status', 'Unknown')
                current_status = item.get('Cuurent Status', 'N/A')
                delivery_date = item.get('Required Delievery Date', 'N/A')
                
                emoji = "❌" if status == "Not Found" else "⚠️" if status == "Insufficient" else "✅"
                
                summary += f"**{i}.** {emoji} **PO:** {po_num}\n"
                summary += f"   **Product:** {product}\n"
                summary += f"   **Delivery Date:** {delivery_date}\n"
                summary += f"   **Status:** {status} - {current_status}\n\n"
            
            if len(data) > 5:
                summary += f"*... and {len(data) - 5} more items*\n\n"
            
            summary += "💡 **Tip:** Ask *'give me in table format'* for a detailed table view"
            
            return summary
            
        except Exception as e:
            logger.error(f"Holding area formatting error: {e}")
            return f"Found {len(data)} items in holding area. Ask for 'table format' for details."
    
    def _format_generic_json_data(self, data: list) -> str:
        """Format generic JSON array data"""
        try:
            summary = f"## 📊 Data Summary\n\n**Total Records:** {len(data)}\n\n"
            
            if len(data) > 0:
                # Show field names
                fields = list(data[0].keys())
                summary += f"**Fields:** {', '.join(fields[:8])}\n"
                if len(fields) > 8:
                    summary += f"*... and {len(fields) - 8} more*\n"
                
                summary += f"\n**Sample Record:**\n"
                for key, value in list(data[0].items())[:6]:
                    summary += f"- **{key}:** {value}\n"
                
                if len(data[0]) > 6:
                    summary += f"*... and {len(data[0]) - 6} more fields*\n"
            
            summary += "\n💡 **Tip:** Ask *'give me in table format'* for a detailed table view"
            
            return summary
            
        except Exception as e:
            logger.error(f"Generic JSON formatting error: {e}")
            return f"Found {len(data)} records. Ask for 'table format' for details."
    
    def close(self):
        """Close the session"""
        if hasattr(self, 'session'):
            self.session.close()
            logger.info("MCP Client session closed")

    def get_available_tools(self) -> Dict[str, Any]:
        """
        Query the MCP server to get list of available tools
        
        Returns:
            Dict containing available tools information
        """
        logger.info("Querying server for available tools...")
        
        for attempt in range(self.max_retries):
            try:
                # Step 1: Establish SSE connection and get session ID
                sse_response = self.session.get(self.sse_url, stream=True, timeout=30)
                
                if sse_response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Failed to establish SSE connection: HTTP {sse_response.status_code}"
                    }
                
                # Step 2: Get session ID and query for tools
                client = sseclient.SSEClient(sse_response)
                session_id = None
                full_response = ""
                start_time = time.time()
                
                for event in client.events():
                    if event.event == "endpoint" and event.data:
                        if "sessionId=" in event.data:
                            session_id = event.data.split("sessionId=")[1]
                            logger.info(f"Got session ID for tools query: {session_id}")
                            
                            # Query for available tools
                            messages_url = f"{self.messages_url}?sessionId={session_id}"
                            
                            payload = {
                                "jsonrpc": "2.0",
                                "id": str(uuid.uuid4()),
                                "method": "tools/list",
                                "params": {}
                            }
                            
                            logger.info(f"Querying tools at: {messages_url}")
                            
                            # Send tools query
                            import threading
                            def send_tools_query():
                                try:
                                    msg_response = self.session.post(messages_url, json=payload, timeout=10)
                                    logger.info(f"Tools query sent, status: {msg_response.status_code}")
                                    if msg_response.status_code != 200:
                                        logger.error(f"Tools query error: {msg_response.text}")
                                except Exception as e:
                                    logger.error(f"Failed to send tools query: {e}")
                            
                            threading.Thread(target=send_tools_query, daemon=True).start()
                            
                    elif event.data and event.data.strip():
                        # Collect response data
                        try:
                            data = json.loads(event.data)
                            if isinstance(data, dict):
                                if 'result' in data and 'tools' in data['result']:
                                    # This is the tools list response
                                    return {
                                        "success": True,
                                        "tools": data['result']['tools']
                                    }
                                elif 'content' in data:
                                    full_response += str(data['content'])
                                elif 'message' in data:
                                    full_response += str(data['message'])
                                else:
                                    full_response += str(data)
                            else:
                                full_response += str(data)
                        except json.JSONDecodeError:
                            full_response += event.data
                        
                        # If we have session and some response, continue listening
                        if session_id and (time.time() - start_time > 15):  # 15 second timeout
                            break
                    
                    # Overall timeout check
                    if time.time() - start_time > 20:
                        break
                
                if full_response.strip():
                    return {
                        "success": True,
                        "response": full_response,
                        "parsed_tools": None
                    }
                else:
                    return {
                        "success": False,
                        "error": "No response received from tools query"
                    }
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Tools query attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": f"Failed to query tools after {self.max_retries} attempts: {str(e)}"
                    }
                    
            except Exception as e:
                logger.error(f"Unexpected error in tools query attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": f"Unexpected error: {str(e)}"
                    }
        
        return {
            "success": False,
            "error": "All retry attempts failed"
        }

# Convenience function for quick queries
def quick_query(message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to send a query without managing client instance
    
    Args:
        message: The message to send
        conversation_id: Optional conversation ID
        
    Returns:
        Dict containing response or error
    """
    client = MCPClient()
    try:
        return client.send_query(message, conversation_id)
    finally:
        client.close() 
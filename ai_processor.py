"""AI Processor for intelligent query handling and data formatting"""

import json
import pandas as pd
from openai import OpenAI
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class AIProcessor:
    """AI-powered processor for handling complex queries and data formatting"""
    
    def __init__(self, api_key: str):
        """
        Initialize AI processor with OpenAI API key
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        
    def can_handle_query(self, query: str, previous_response: Optional[Dict] = None) -> bool:
        """
        Determine if this query needs AI processing
        
        Args:
            query: User query
            previous_response: Previous response data
            
        Returns:
            True if AI should handle this query
        """
        query_lower = query.lower()
        
        # AI should handle if:
        ai_keywords = [
            'table format', 'format', 'summarize', 'summary', 'analyze', 
            'how many', 'count', 'total', 'average', 'which', 'what',
            'show me', 'filter', 'sort', 'group', 'compare', 'explain'
        ]
        
        # Or if there's previous data to work with
        has_previous_data = previous_response and 'result' in previous_response
        has_ai_keywords = any(keyword in query_lower for keyword in ai_keywords)
        
        return has_ai_keywords or has_previous_data
    
    def process_mcp_response(self, user_query: str, mcp_response: Dict, 
                           conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Process MCP server response and format it like Claude Desktop
        
        Args:
            user_query: The original user query
            mcp_response: Raw response from MCP server
            conversation_context: Recent conversation context
            
        Returns:
            Formatted response for user display
        """
        try:
            logger.info(f"Processing MCP response for query: '{user_query}'")
            
            # Extract JSON data from MCP response
            extracted_data = self._extract_json_from_mcp_response(mcp_response)
            
            if extracted_data:
                logger.info(f"Successfully extracted {len(extracted_data)} items, formatting with OpenAI...")
                # Format the data using OpenAI (like Claude Desktop does)
                formatted_response = self._format_data_with_openai(user_query, extracted_data, conversation_context)
                
                logger.info(f"OpenAI formatted response: {formatted_response[:200]}...")
                return {
                    "success": True,
                    "response": formatted_response,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "source": "openai_formatted"
                }
            else:
                logger.warning("No structured data extracted, using cleanup method")
                # No structured data found, just clean up the response
                raw_text = str(mcp_response.get('response', ''))
                cleaned_response = self._clean_raw_response(raw_text)
                
                logger.info(f"Cleaned response: {cleaned_response[:200]}...")
                return {
                    "success": True,
                    "response": cleaned_response,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "source": "cleaned_text"
                }
                
        except Exception as e:
            logger.error(f"MCP response processing error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to process response: {str(e)}",
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def process_query(self, query: str, previous_response: Optional[Dict] = None, 
                     conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Process query using AI
        
        Args:
            query: User query
            previous_response: Previous MCP response data
            conversation_context: Recent conversation messages
            
        Returns:
            Processed response
        """
        try:
            # Prepare context
            context = self._prepare_context(query, previous_response, conversation_context)
            
            # Get AI response
            ai_response = self._get_ai_response(context)
            
            # Format response
            formatted_response = self._format_ai_response(ai_response, previous_response)
            
            return {
                "success": True,
                "response": formatted_response,
                "timestamp": pd.Timestamp.now().isoformat(),
                "source": "ai_processor"
            }
            
        except Exception as e:
            logger.error(f"AI processing error: {str(e)}")
            return {
                "success": False,
                "error": f"AI processing failed: {str(e)}",
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def _prepare_context(self, query: str, previous_response: Optional[Dict], 
                        conversation_context: List[Dict]) -> str:
        """Prepare context for AI"""
        
        context_parts = [
            "You are an intelligent assistant for BSL inventory company.",
            "You help with inventory management, delivery orders (DO), purchase orders (PO), and data analysis.",
            "",
            "Company abbreviations:",
            "- DO(s): Delivery Order(s)",
            "- PO(s): Purchase Order(s)", 
            "- PR(s): Product Request(s)",
            "- QA: Quality Assessment",
            "- NCR: Non-Conformance Report",
            "",
            f"Current user query: {query}",
            ""
        ]
        
        # Add previous data if available
        if previous_response and 'result' in previous_response:
            context_parts.extend([
                "Previous data retrieved:",
                json.dumps(previous_response['result'], indent=2),
                ""
            ])
        
        # Add conversation context
        if conversation_context:
            context_parts.append("Recent conversation:")
            for msg in conversation_context[-3:]:  # Last 3 messages
                role = "User" if msg.get('is_user') else "Assistant"
                content = msg.get('content', '')[:200]  # Truncate long messages
                context_parts.append(f"{role}: {content}")
            context_parts.append("")
        
        context_parts.extend([
            "Instructions:",
            "1. If user asks for table format, convert JSON data to a readable table",
            "2. For analysis queries (count, total, etc.), calculate from the data",
            "3. Be concise but informative",
            "4. Use markdown formatting for tables",
            "5. If data formatting is requested, create clean, organized output",
            "",
            "Respond with the requested information:"
        ])
        
        return "\n".join(context_parts)
    
    def _get_ai_response(self, context: str) -> str:
        """Get response from OpenAI"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Using more cost-effective model
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant for BSL inventory company. Format data clearly and provide concise, actionable responses."
                },
                {"role": "user", "content": context}
            ],
            max_tokens=2000,
            temperature=0.1  # Low temperature for consistent formatting
        )
        
        return response.choices[0].message.content
    
    def _format_ai_response(self, ai_response: str, previous_response: Optional[Dict]) -> str:
        """Format AI response for display"""
        
        # If the AI response contains a table and we have JSON data, 
        # try to enhance it with actual pandas DataFrame formatting
        if previous_response and "table" in ai_response.lower():
            try:
                data = self._extract_tabular_data(previous_response)
                if data:
                    df = pd.DataFrame(data)
                    # Add DataFrame info to response
                    table_info = f"\n\n**Data Summary:**\n- Total rows: {len(df)}\n- Columns: {', '.join(df.columns)}"
                    return ai_response + table_info
            except Exception as e:
                logger.warning(f"Could not enhance table format: {e}")
        
        return ai_response
    
    def _extract_tabular_data(self, response: Dict) -> Optional[List[Dict]]:
        """Extract tabular data from MCP response"""
        
        try:
            if 'result' in response and 'content' in response['result']:
                content = response['result']['content']
                if isinstance(content, list) and content:
                    text_content = content[0].get('text', '')
                    
                    # Try to parse JSON from text
                    if text_content.startswith('[') or text_content.startswith('{'):
                        data = json.loads(text_content)
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict):
                            return [data]
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not extract tabular data: {e}")
            return None
    
    def format_json_as_table(self, json_data: str) -> str:
        """Convert JSON data to markdown table format"""
        
        try:
            # Import tabulate here to avoid dependency issues
            from tabulate import tabulate
            
            data = json.loads(json_data)
            if not isinstance(data, list) or not data:
                return "No tabular data found."
            
            df = pd.DataFrame(data)
            
            # Limit columns for readability
            if len(df.columns) > 8:
                # Show most important columns for inventory data
                important_cols = ['POnumber', 'productCode', 'Ordered Date', 'Required Delievery Date', 
                                'Ordered Quantity', 'Current QOH', 'Stock Status', 'Cuurent Status']
                display_cols = [col for col in important_cols if col in df.columns]
                if len(display_cols) < 5:  # If not enough important columns, take first few
                    display_cols = df.columns[:6].tolist()
                df = df[display_cols]
            
            # Create markdown table using tabulate
            table_md = tabulate(df, headers='keys', tablefmt='github', showindex=False, maxcolwidths=[20]*len(df.columns))
            
            # Additional insights
            insights = []
            if 'Stock Status' in df.columns:
                status_counts = df['Stock Status'].value_counts()
                for status, count in status_counts.items():
                    emoji = "❌" if status == "Not Found" else "⚠️" if status == "Insufficient" else "✅"
                    insights.append(f"- {emoji} {status}: {count} items")
            
            if 'POnumber' in df.columns:
                unique_pos = df['POnumber'].nunique()
                insights.append(f"- 📋 Unique POs: {unique_pos}")
            
            if 'Required Delievery Date' in df.columns:
                overdue_count = 0
                try:
                    df['delivery_date_parsed'] = pd.to_datetime(df['Required Delievery Date'], errors='coerce')
                    overdue_count = len(df[df['delivery_date_parsed'] < pd.Timestamp.now()])
                    if overdue_count > 0:
                        insights.append(f"- 🚨 Overdue items: {overdue_count}")
                except:
                    pass
            
            summary = f"""## 📊 Holding Area Data Table

**Summary:** {len(df)} items found

{table_md}

**Key Insights:**
{chr(10).join(insights) if insights else '- No specific insights available'}

---
*💡 Tip: You can ask for specific analysis like "show me overdue items" or "group by supplier"*
"""
            
            return summary
            
        except ImportError as e:
            logger.error(f"Tabulate not available: {e}")
            # Fallback to pandas markdown if tabulate not available
            try:
                df = pd.DataFrame(json.loads(json_data))
                return f"## 📊 Data Table\n\n{df.to_markdown(index=False)}\n\n*Note: Install tabulate for better formatting*"
            except:
                return f"Could not format as table: Missing tabulate dependency. Run: pip install tabulate"
        except Exception as e:
            logger.error(f"Table formatting error: {e}")
            return f"Could not format as table: {str(e)}"
    
    def _extract_json_from_mcp_response(self, mcp_response: Dict) -> Optional[List[Dict]]:
        """Extract JSON data from MCP response structure"""
        try:
            logger.info(f"Extracting JSON from MCP response: {str(mcp_response)[:200]}...")
            
            # Method 1: Try direct dictionary access to 'response' field
            if 'response' in mcp_response:
                response_data = mcp_response['response']
                logger.info(f"Found response field: {str(response_data)[:200]}...")
                
                # Handle the triple-nested structure: {'content': [{'type': 'text', 'text': '[{...}]'}]}
                if isinstance(response_data, dict) and 'content' in response_data:
                    content = response_data['content']
                    logger.info(f"Found content field: {str(content)[:200]}...")
                    
                    if isinstance(content, list) and len(content) > 0:
                        first_item = content[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            text_data = first_item['text']
                            logger.info(f"Found text data: {str(text_data)[:200]}...")
                            
                            # Try to parse the triple-nested JSON structure
                            try:
                                # First level: Parse the outer JSON array
                                outer_data = json.loads(text_data)
                                logger.info(f"Parsed outer level, got: {type(outer_data)} with {len(outer_data) if isinstance(outer_data, list) else 'N/A'} items")
                                
                                if isinstance(outer_data, list) and len(outer_data) > 0:
                                    inner_item = outer_data[0]
                                    logger.info(f"Inner item: {str(inner_item)[:200]}...")
                                    
                                    if isinstance(inner_item, dict) and 'text' in inner_item:
                                        inner_text = inner_item['text']
                                        logger.info(f"Found inner text: {str(inner_text)[:200]}...")
                                        
                                        # Second level: Parse the actual inventory data
                                        actual_data = json.loads(inner_text)
                                        logger.info(f"Successfully extracted {len(actual_data)} items from triple-nested structure")
                                        return actual_data
                                        
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse triple-nested JSON: {e}")
                                
                                # Fallback: Try to extract with regex from the text_data
                                import re
                                json_match = re.search(r'\[{"row_number".*?}\]', text_data, re.DOTALL)
                                if json_match:
                                    try:
                                        json_str = json_match.group(0)
                                        # Clean up escaped characters
                                        json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
                                        fallback_data = json.loads(json_str)
                                        logger.info(f"Fallback regex extraction successful: {len(fallback_data)} items")
                                        return fallback_data
                                    except Exception as regex_e:
                                        logger.warning(f"Regex fallback also failed: {regex_e}")
            
            # Method 2: String-based regex extraction as fallback
            response_text = str(mcp_response)
            import re
            
            # Look for deeply nested pattern: "text":"[{...}]"
            nested_pattern = r'"text":"(\[{.*?}\])"'
            nested_match = re.search(nested_pattern, response_text, re.DOTALL)
            
            if nested_match:
                json_str = nested_match.group(1)
                # Unescape the JSON
                json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
                logger.info(f"Found nested pattern, trying to parse: {json_str[:200]}...")
                
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"Successfully parsed {len(data)} items via regex")
                        return data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse regex-extracted JSON: {e}")
            
            logger.warning("No JSON data could be extracted from MCP response")
            return None
            
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            return None
    
    def _format_data_with_openai(self, user_query: str, data: List[Dict], 
                               conversation_context: List[Dict] = None) -> str:
        """Use OpenAI to format data like Claude Desktop"""
        try:
            # Prepare context for OpenAI
            context = self._prepare_formatting_context(user_query, data, conversation_context)
            
            # Get formatted response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that formats BSL inventory data for users. 
                        
Your job is to take raw inventory data and present it in a clear, human-readable format similar to how Claude Desktop would present it.

Key guidelines:
1. Always start with a brief summary of what was found
2. Present data in an organized, scannable format
3. Use clear headings and bullet points
4. Highlight important information like stock status
5. Be concise but informative
6. Use markdown formatting for better readability
7. Focus on the most relevant information for the user's query

Make the output professional but easy to read at a glance."""
                    },
                    {"role": "user", "content": context}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI formatting error: {e}")
            # Fallback to basic formatting
            return self._basic_data_formatting(data)
    
    def _prepare_formatting_context(self, user_query: str, data: List[Dict], 
                                  conversation_context: List[Dict] = None) -> str:
        """Prepare context for OpenAI formatting"""
        
        context_parts = [
            f"User asked: '{user_query}'",
            "",
            f"Raw data retrieved ({len(data)} items):",
            json.dumps(data, indent=2),
            "",
            "Please format this data in a clear, human-readable way that directly answers the user's question.",
            "Focus on the most important information and present it professionally."
        ]
        
        # Add conversation context if available
        if conversation_context:
            context_parts.insert(-2, "Recent conversation context:")
            for msg in conversation_context[-2:]:  # Last 2 messages
                role = "User" if msg.get('is_user') else "Assistant"
                content = msg.get('content', '')[:100]  # Truncate
                context_parts.insert(-2, f"{role}: {content}")
            context_parts.insert(-2, "")
        
        return "\n".join(context_parts)
    
    def _clean_raw_response(self, raw_text: str) -> str:
        """Clean up raw response text"""
        if not raw_text:
            return "No data received from server."
        
        # If this is still showing raw JSON structure, let's try to force OpenAI formatting
        if '"content"' in raw_text and '"text"' in raw_text:
            logger.info("Detected raw JSON structure, attempting forced OpenAI formatting...")
            
            # Try to extract data one more time with a simpler approach
            import re
            json_match = re.search(r'\[{.*?}\]', raw_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Clean up escaped characters
                    json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
                    data = json.loads(json_str)
                    
                    if isinstance(data, list) and len(data) > 0:
                        # Force OpenAI formatting as last resort
                        logger.info(f"Forcing OpenAI formatting for {len(data)} items...")
                        return self._basic_data_formatting(data)
                        
                except Exception as e:
                    logger.warning(f"Final extraction attempt failed: {e}")
        
        # Basic cleanup
        cleaned = raw_text.strip()
        
        # If it looks like JSON, try to make it more readable
        if cleaned.startswith('{') or cleaned.startswith('['):
            try:
                # Try to parse and reformat JSON
                data = json.loads(cleaned)
                return json.dumps(data, indent=2)
            except:
                pass
        
        return cleaned
    
    def _basic_data_formatting(self, data: List[Dict]) -> str:
        """Basic fallback formatting"""
        if not data:
            return "No data found."
        
        summary = f"## Data Summary\n\n**Total Items:** {len(data)}\n\n"
        
        # Show first few items
        for i, item in enumerate(data[:5], 1):
            summary += f"**Item {i}:**\n"
            for key, value in list(item.items())[:6]:
                summary += f"- {key}: {value}\n"
            summary += "\n"
        
        if len(data) > 5:
            summary += f"*... and {len(data) - 5} more items*"
        
        return summary 
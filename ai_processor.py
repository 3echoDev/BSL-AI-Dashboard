"""AI Processor for intelligent query handling and data formatting"""

import json
import pandas as pd
from anthropic import Anthropic
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class AIProcessor:
    """AI-powered processor for handling complex queries and data formatting"""
    
    def __init__(self, api_key: str):
        """
        Initialize AI processor with Claude API key
        
        Args:
            api_key: Claude API key
        """
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        
    def can_handle_query(self, query: str, previous_response: Optional[Dict] = None, tool_info: Optional[Dict] = None) -> bool:
        """
        Determine if this query needs AI processing
        
        Args:
            query: User query
            previous_response: Previous response data
            tool_info: Information about which tool will handle the query
            
        Returns:
            True if AI should handle this query
        """
        query_lower = query.lower()
        
        # AI should handle if:
        ai_keywords = [
            'table format', 'format', 'summarize', 'summary', 'analyze', 
            'how many', 'count', 'total', 'average', 'which', 'what',
            'show me', 'filter', 'sort', 'group', 'compare', 'explain',
            'table', 'tabular', 'in table', 'as table', 'format as table',
            'comprehensive', 'detailed analysis', 'full analysis', 'report'
        ]
        
        # Or if there's previous data to work with
        has_previous_data = previous_response and 'result' in previous_response
        has_ai_keywords = any(keyword in query_lower for keyword in ai_keywords)
        
        # Special handling for NCR queries which often have large datasets
        is_ncr_query = any(ncr_keyword in query_lower for ncr_keyword in ['ncr', 'non-conformance', 'non conformance'])
        
        # Agent workflow tools should generally use AI formatting
        is_agent_workflow = tool_info and tool_info.get('type') == 'agent_workflow'
        
        # Record agent tools with analysis requests should use AI
        is_analysis_request = tool_info and tool_info.get('type') == 'record_agent' and has_ai_keywords
        
        return has_ai_keywords or has_previous_data or is_ncr_query or is_agent_workflow or is_analysis_request
    
    def process_mcp_response(self, user_query: str, mcp_response: Dict, 
                           conversation_context: List[Dict] = None, tool_context: Dict = None) -> Dict[str, Any]:
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
                logger.info(f"Successfully extracted {len(extracted_data)} items, formatting with Claude...")
                
                # Extract nested JSON from result fields if needed
                extracted_data = self._extract_nested_results(extracted_data)
                
                # Check if user specifically requested table format
                is_table_request = any(keyword in user_query.lower() for keyword in [
                    'table', 'tabular', 'format as table', 'in table format', 'table format'
                ])
                
                if is_table_request:
                    # Use specialized table format handler
                    formatted_response = self.process_table_format_request(user_query, extracted_data, conversation_context, tool_context)
                else:
                    # Format the data using Claude (like Claude Desktop does)
                    formatted_response = self._format_data_with_claude(user_query, extracted_data, conversation_context, tool_context)
                
                logger.info(f"Claude formatted response: {formatted_response[:200]}...")
                return {
                    "success": True,
                    "response": formatted_response,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "source": "claude_formatted"
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
        Process query using AI with enhanced memory for "more records" requests
        
        Args:
            query: User query
            previous_response: Previous MCP response data
            conversation_context: Recent conversation messages
            
        Returns:
            Processed response
        """
        try:
            query_lower = query.lower()
            
            # Check if this is a "more records" type request
            is_more_request = any(keyword in query_lower for keyword in [
                'more records', 'more data', 'give me more', 'show more', 'continue', 'next'
            ])
            
            if is_more_request and previous_response:
                logger.info("Processing 'more records' request with stored data")
                # Extract data from stored response
                extracted_data = self._extract_json_from_mcp_response(previous_response)
                
                if extracted_data:
                    # Extract nested results if needed
                    extracted_data = self._extract_nested_results(extracted_data)
                    
                    # Create enhanced context for showing more records
                    enhanced_query = f"Show more records from the NCR dataset. Display a comprehensive table with 50+ actual records showing real NCR numbers, part numbers, dates, and statuses. Original query: {query}"
                    
                    # Use table format handler for better display
                    formatted_response = self.process_table_format_request(
                        enhanced_query, 
                        extracted_data, 
                        conversation_context,
                        {'tool': 'Check_full_ncr', 'type': 'record_agent'}
                    )
                    
                    return {
                        "success": True,
                        "response": formatted_response,
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "source": "ai_processor_memory"
                    }
            
            # Prepare context for regular queries
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
        
        # Add previous data if available (truncate if too large)
        if previous_response and 'result' in previous_response:
            result_data = previous_response['result']
            if isinstance(result_data, (list, dict)):
                # For large datasets, show summary instead of full data
                if isinstance(result_data, list) and len(result_data) > 50:
                    sample_data = result_data[:10]  # Show first 10 items
                    context_parts.extend([
                        f"Previous data retrieved (showing 10 of {len(result_data)} items):",
                        json.dumps(sample_data, indent=2),
                        f"... and {len(result_data) - 10} more items",
                        ""
                    ])
                else:
                    context_parts.extend([
                        "Previous data retrieved:",
                        str(result_data)[:2000] + "..." if len(str(result_data)) > 2000 else str(result_data),
                        ""
                    ])
            else:
                context_parts.extend([
                    "Previous data retrieved:",
                    str(result_data)[:2000] + "..." if len(str(result_data)) > 2000 else str(result_data),
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
            "1. If user asks for table format, provide comprehensive table analysis like Claude Desktop",
            "2. For analysis queries (count, total, etc.), calculate from the data",
            "3. For large datasets, provide professional business report format",
            "4. Use markdown formatting for tables and structure",
            "5. Include sample records, key metrics, and actionable insights",
            "6. Detect data type (NCR, PO, DO, etc.) and provide specialized analysis",
            "",
            "Respond with the requested information:"
        ])
        
        return "\n".join(context_parts)
    
    def _get_ai_response(self, context: str) -> str:
        """Get response from Claude"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Using Claude-3.5-Sonnet like Claude Desktop
            max_tokens=8000,  # Increased token limit for large datasets
            temperature=0.1,  # Low temperature for consistent formatting
            system="You are a helpful assistant for BSL inventory company. Format data clearly and provide concise, actionable responses. Handle large datasets effectively.",
            messages=[
                {"role": "user", "content": context}
            ]
        )
        
        return response.content[0].text
    
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
    
    def format_json_as_table(self, json_data: str, tool_context: Dict = None) -> str:
        """Convert JSON data to comprehensive table format like Claude Desktop"""
        
        try:
            data = json.loads(json_data)
            if not isinstance(data, list) or not data:
                return "No tabular data found."
            
            # Use Claude to create comprehensive table analysis
            return self._create_comprehensive_table_analysis(data, tool_context)
            
        except Exception as e:
            logger.error(f"Table formatting error: {e}")
            return f"Could not format as table: {str(e)}"
    
    def _create_comprehensive_table_analysis(self, data: List[Dict], tool_context: Dict = None) -> str:
        """Create comprehensive table analysis like Claude Desktop"""
        try:
            df = pd.DataFrame(data)
            
            # Use tool context to determine data type, fallback to detection
            if tool_context:
                logger.info(f"🔧 TOOL CONTEXT RECEIVED: {tool_context}")
                if 'tool' in tool_context:
                    data_type = self._get_data_type_from_tool(tool_context['tool'])
                elif 'tool_name' in tool_context:
                    data_type = self._get_data_type_from_tool(tool_context['tool_name'])
                else:
                    logger.warning(f"⚠️ Tool context missing 'tool' field: {tool_context}")
                    data_type = self._detect_data_type(df)
            else:
                logger.info("🔍 No tool context provided, using fallback detection")
                data_type = self._detect_data_type(df)
            
            # Set maximum allowed token limit for Claude
            max_tokens = 8192  # Maximum allowed for Claude 3.5 Sonnet
            large_dataset_note = ""  # No truncation notes
            
            # Create context for Claude to generate comprehensive analysis
            # Create specific context based on data type
            if data_type == "Holding Area":
                # For large datasets, provide structured sample + instructions
                if len(data) > 30:
                    sample_data = data[:80]  # Show first 80 for more records display
                    
                    # Clean up column names
                    cleaned_sample = []
                    for record in sample_data:
                        cleaned_record = {}
                        for key, value in record.items():
                            clean_key = key.strip()
                            cleaned_record[clean_key] = value
                        cleaned_sample.append(cleaned_record)
                    
                    dataset_info = f"First 80 records from {len(data)} total:\n{json.dumps(cleaned_sample, indent=2)}\n\nIMPORTANT: Display 50+ actual data rows in the table with real values."
                else:
                    dataset_info = json.dumps(data, indent=2)
                
                analysis_context = f"""Create a comprehensive Holding Area report for BSL inventory.

Dataset: {len(data)} items in holding area
Columns: {', '.join(df.columns.tolist())}

Complete dataset for table generation:
{dataset_info}

Full column statistics:
{self._generate_column_statistics(df)}

CRITICAL INSTRUCTIONS for Holding Area data:
1. **Title should be "BSL Holding Area Report"** - NOT Purchase Order
2. **This is HOLDING AREA data** - items waiting for processing/approval
3. **SHOW COMPLETE DATA TABLE** - Include ALL {len(data)} items in the table (do NOT truncate unless 100+ items)
4. **NO "[...remaining X items...]"** - Show every single row of data
5. **Focus on**: Stock Status, Current Status, Required Delivery Dates, PIC assignments
6. **Highlight urgent items**: Items with "Not Found" or "Insufficient" stock status
7. **Show delivery timeline**: Group by delivery dates to show urgency
8. **PIC workload**: Show distribution of items per person responsible
9. **Action items**: Focus on supplier sourcing and stock replenishment

Format as: BSL Holding Area Management Report with actionable insights for operations team.
IMPORTANT: Show the COMPLETE table with all {len(data)} rows - users need to see every item.{large_dataset_note}"""
            else:
                # For large datasets, provide structured approach
                if len(data) > 30:
                    # Create a more efficient representation - send enough data for 50+ rows
                    sample_records = data[:80]  # First 80 records to ensure Claude can show 50+
                    
                    # Convert to a more compact format for Claude
                    compact_data = []
                    for record in sample_records:
                        compact_data.append(record)
                    
                    # Also create a table preview to help Claude understand the format
                    try:
                        df_preview = pd.DataFrame(sample_records)
                        table_preview = df_preview.to_string(index=False, max_rows=10)
                    except Exception as e:
                        logger.warning(f"Could not create table preview: {e}")
                        table_preview = "Table preview not available"
                    
                    # Clean up column names for better presentation
                    cleaned_sample = []
                    for record in sample_records:
                        cleaned_record = {}
                        for key, value in record.items():
                            # Clean up column names (remove extra spaces)
                            clean_key = key.strip()
                            cleaned_record[clean_key] = value
                        cleaned_sample.append(cleaned_record)
                    
                    # Get column structure for Claude
                    if cleaned_sample:
                        column_list = list(cleaned_sample[0].keys())
                        column_info = f"Columns in order: {', '.join(column_list)}"
                    else:
                        column_info = "No column information available"
                    
                    dataset_info = f"""Dataset contains {len(data)} NCR records. 

{column_info}

First 80 records for table generation:
{json.dumps(cleaned_sample, indent=2)}

Table format preview:
{table_preview}

CRITICAL INSTRUCTIONS FOR DISPLAYING MORE RECORDS:
1. **DISPLAY 50+ ACTUAL DATA ROWS** - Show at least 50 real records in the table
2. **USE THE PROVIDED DATA** - Display records from the 80 records provided above
3. **PROPER COLUMN ORDER**: S/N, NCR Number, Part Number, Part Name, Customer, Date, Issue Description, Qty, Disposition, Status
4. **REAL DATA ONLY** - Show actual NCR numbers (NCR0001, NCR0002, etc.), real part numbers, real dates
5. **NO TRUNCATION NOTES** - Don't show "First X records" - just display the actual data table
6. **CLEAN FORMATTING** - Use proper markdown table with aligned columns

GOAL: Show a comprehensive table with 50+ actual NCR records so users can see substantial data."""
                else:
                    dataset_info = json.dumps(data, indent=2)
                
                analysis_context = f"""Create a comprehensive table format analysis for this {data_type} dataset.

Dataset: {len(data)} items
Columns: {', '.join(df.columns.tolist())}

Complete dataset for table generation:
{dataset_info}

Full column statistics:
{self._generate_column_statistics(df)}

Instructions for formatting BSL business data:
1. **SHOW ACTUAL DATA ROWS** - Display as many real data rows as possible (aim for 50+ if space allows)
2. **NO TRUNCATION MESSAGES** - Never show "[Table truncated for brevity]" - show actual data instead
3. **REAL RECORDS ONLY** - Show actual NCR numbers, part numbers, dates, etc. from the dataset
4. **Use clear, business-friendly headers** 
5. **For DO/PO data**: Focus on order status, delivery dates, quantities, and stock status
6. **For NCR data**: Show NCR numbers, part numbers, dates, issues, disposition, status
7. **For inventory data**: Show quantities, locations, and stock levels
8. **Include summary statistics** after the table
9. **Use proper markdown table formatting**
10. **Highlight critical information** (overdue, insufficient stock, high priority)
11. **Make it scannable for business users**

Format like a professional business report with:
- DATA TABLE FIRST with actual records (show 50+ rows if possible, never just headers)
- Key insights and metrics
- Status summaries  
- Action items if relevant

CRITICAL: Show REAL DATA ROWS from the {len(data)} records - not placeholders or truncation messages.
If you can't show all {len(data)} rows due to space, show as many as possible (50+ preferred) with real data."""
            

            
            # Get comprehensive analysis from Claude
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=0.1,
                system=f"""You are creating a comprehensive NCR table analysis for BSL like Claude Desktop does.

CRITICAL FORMATTING RULES:
1. **SHOW 50+ REAL DATA ROWS** - Display at least 50 actual records with real data (MANDATORY)
2. **CLEAN TABLE STRUCTURE** - Use proper markdown table formatting with aligned columns
3. **LOGICAL COLUMN ORDER** - Follow this order: S/N, NCR Number, Part Number, Part Name, Customer, Date, Issue Description, Qty, Disposition, Status
4. **ACTUAL NCR DATA** - Show real NCR numbers (NCR0001, NCR0002, NCR0003, etc.), real part numbers, real dates
5. **NO SUMMARY ONLY** - Don't just show "First X records" - display the actual full table with 50+ rows
6. **CLEAN HEADERS** - Use business-friendly column names without extra spaces

DISPLAY REQUIREMENT - SHOW 50+ ROWS LIKE THIS:
| S/N | NCR Number | Part Number | Part Name | Customer | Date | Issue Description | Qty | Disposition | Status |
|-----|------------|-------------|-----------|----------|------|-------------------|-----|-------------|--------|
| 1   | NCR0001    | 09545-0107-025-02 | DVI AND AUDIO SPLITTER MOUNT | K&S | 27-Dec-21 | Welding/buffing issues | 14 | REWORK | CLOSED |
| 2   | NCR0002    | 09545-0107-012-04 | UPPER HEAT EXCHANGER TUBE TRAY | K&S | 27-Dec-21 | Surface issues | 14 | REWORK | CLOSED |
[...continue for 50+ rows with real data...]

CONTENT REQUIREMENTS:
1. **Professional title**: "BSL Non-Conformance Report (NCR) Analysis"
2. **FULL DATA TABLE**: Show 50+ actual records (NOT a summary) - display real NCR data rows
3. **Summary statistics**: Count by status, customer breakdown, etc. (AFTER the table)
4. **Key insights**: Highlight patterns and issues (AFTER the table)

MANDATORY: Display a table with 50+ actual data rows from the provided dataset. Users need to see substantial data, not just summaries.""",
                messages=[
                    {"role": "user", "content": analysis_context}
                ]
            )
            
            formatted_analysis = response.content[0].text
            
            # Ensure we have a proper table - if Claude didn't include it, add it
            try:
                from tabulate import tabulate
                
                # Check if the response already has a markdown table
                if "|" not in formatted_analysis or "---" not in formatted_analysis:
                    # Add the table at the beginning if missing
                    table_md = tabulate(df, headers='keys', tablefmt='github', showindex=False, maxcolwidths=[25]*len(df.columns))
                    formatted_analysis = f"## 📊 **{data_type} Data**\n\n{table_md}\n\n" + formatted_analysis
                
            except ImportError:
                # Fallback without tabulate
                if "|" not in formatted_analysis:
                    formatted_analysis = f"## 📊 **{data_type} Data**\n\n{df.to_markdown(index=False)}\n\n" + formatted_analysis
            
            return formatted_analysis
            
        except Exception as e:
            logger.error(f"Comprehensive table analysis error: {e}")
            return self._fallback_table_format(data)
    
    def _get_data_type_from_tool(self, tool_name: str) -> str:
        """Get data type based on the actual tool that was called"""
        tool_mappings = {
            'Check_DO': 'Delivery Order (DO)',
            'Check_Holding_Area': 'Holding Area', 
            'Check_Full_Treatment': 'Treatment Parts',
            'Check_full_ncr': 'Non-Conformance Report (NCR)',
            # Agent workflow tools
            'Agent': 'Business Data'
        }
        
        data_type = tool_mappings.get(tool_name, 'Business Data')
        logger.info(f"🎯 TOOL-BASED DATA TYPE: Tool '{tool_name}' → Data type '{data_type}'")
        return data_type
    
    def _detect_data_type(self, df: pd.DataFrame) -> str:
        """Detect the type of data for specialized formatting (fallback only)"""
        columns = [col.lower() for col in df.columns]
        actual_columns = df.columns.tolist()
        
        logger.info(f"🔍 FALLBACK DATA TYPE DETECTION: Columns found: {actual_columns}")
        
        # More conservative detection - only when very specific
        if any('holding' in col for col in columns):
            logger.info("✅ DETECTED: Holding Area (holding keyword)")
            return "Holding Area"
        elif any('ncr' in col for col in columns):
            logger.info("✅ DETECTED: Non-Conformance Report (NCR)")
            return "Non-Conformance Report (NCR)"
        elif any('delivery' in col for col in columns):
            logger.info("✅ DETECTED: Delivery Order (DO)")
            return "Delivery Order (DO)"
        elif any('treatment' in col for col in columns):
            logger.info("✅ DETECTED: Treatment Parts")
            return "Treatment Parts"
        else:
            logger.info("⚠️ DETECTED: Generic Business Data")
            return "Business Data"
    
    def _generate_column_statistics(self, df: pd.DataFrame) -> str:
        """Generate statistics for each column"""
        stats = []
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                most_common = df[col].value_counts().head(3)
                stats.append(f"- {col}: {unique_count} unique values, most common: {dict(most_common)}")
            else:
                stats.append(f"- {col}: numeric data, range: {df[col].min()} to {df[col].max()}")
        
        return "\n".join(stats[:10])  # Limit to avoid token overflow
    
    def _fallback_table_format(self, data: List[Dict]) -> str:
        """Fallback table formatting if comprehensive analysis fails"""
        try:
            from tabulate import tabulate
            df = pd.DataFrame(data)
            table_md = tabulate(df, headers='keys', tablefmt='github', showindex=False, maxcolwidths=[20]*len(df.columns))
            
            return f"""## 📊 Data Table Format

**Total Records:** {len(data)}

{table_md}

**Quick Analysis:**
- Dataset contains {len(data)} records
- Available columns: {', '.join(df.columns.tolist())}
- Data type: {self._detect_data_type(df)}

**Working with this data:**
- Filter by any column value
- Request specific analysis
- Export to different formats
- Ask for insights on patterns

*💡 Ask for specific analysis or filtering for more insights!*
"""
        except Exception:
            return f"**Data Summary:** {len(data)} records available. Ask for specific formatting or analysis."
    
    def _extract_json_from_mcp_response(self, mcp_response: Dict) -> Optional[List[Dict]]:
        """Extract JSON data from MCP response structure - Enhanced for Agent workflow responses"""
        try:
            logger.info(f"Extracting JSON from MCP response: {str(mcp_response)[:200]}...")
            
            # Method 1: Try direct dictionary access to 'response' field
            if 'response' in mcp_response:
                response_data = mcp_response['response']
                logger.info(f"Found response field: {str(response_data)[:200]}...")
                
                # Handle Agent workflow response structure: {'content': [{'type': 'text', 'text': '[{"output": "..."}]'}]}
                if isinstance(response_data, dict) and 'content' in response_data:
                    content = response_data['content']
                    logger.info(f"Found content field: {str(content)[:200]}...")
                    
                    if isinstance(content, list) and len(content) > 0:
                        first_item = content[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            text_data = first_item['text']
                            logger.info(f"Found text data: {str(text_data)[:200]}...")
                            
                            # NEW: Handle Agent workflow "output" structure
                            try:
                                parsed_data = json.loads(text_data)
                                logger.info(f"Parsed data: {type(parsed_data)}")
                                
                                # Check if it's Agent workflow format: [{"output": "..."}]
                                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                    first_parsed_item = parsed_data[0]
                                    if isinstance(first_parsed_item, dict) and 'output' in first_parsed_item:
                                        logger.info("🤖 Detected Agent workflow 'output' structure")
                                        # Convert to a format that Claude can work with
                                        formatted_data = [{
                                            "result": first_parsed_item['output'],
                                            "type": "agent_workflow_response",
                                            "timestamp": pd.Timestamp.now().isoformat()
                                        }]
                                        return formatted_data
                                    
                                    # Check if it's triple-nested structure
                                    if isinstance(first_parsed_item, dict) and 'text' in first_parsed_item:
                                        inner_text = first_parsed_item['text']
                                        logger.info(f"Found inner text: {str(inner_text)[:200]}...")
                                        
                                        # Second level: Parse the actual inventory data
                                        actual_data = json.loads(inner_text)
                                        logger.info(f"Successfully extracted {len(actual_data)} items from triple-nested structure")
                                        
                                        # Third level: Check if each item has a "result" field with JSON data
                                        if (len(actual_data) > 0 and 
                                            isinstance(actual_data[0], dict) and 
                                            'result' in actual_data[0] and 
                                            isinstance(actual_data[0]['result'], str)):
                                            
                                            logger.info("🔄 Detected nested JSON in 'result' fields, extracting...")
                                            extracted_records = []
                                            
                                            for item in actual_data:
                                                if 'result' in item and isinstance(item['result'], str):
                                                    try:
                                                        # Parse the JSON string inside the result field
                                                        nested_record = json.loads(item['result'])
                                                        extracted_records.append(nested_record)
                                                    except json.JSONDecodeError as e:
                                                        logger.warning(f"Failed to parse nested result JSON: {e}")
                                                        # Keep the original item if parsing fails
                                                        extracted_records.append(item)
                                                else:
                                                    extracted_records.append(item)
                                            
                                            logger.info(f"✅ Successfully extracted {len(extracted_records)} NCR records from nested results")
                                            return extracted_records
                                        
                                        return actual_data
                                    
                                    # Check if it's standard data array
                                    if isinstance(first_parsed_item, dict) and any(key in first_parsed_item for key in ['row_number', 'POnumber', 'productCode']):
                                        logger.info(f"Successfully extracted {len(parsed_data)} items from standard array")
                                        return parsed_data
                                        
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON structure: {e}")
                                
                                # Enhanced regex fallback for Agent workflow responses
                                import re
                                
                                # Look for Agent workflow output pattern
                                output_pattern = r'"output":\s*"([^"]*)"'
                                output_match = re.search(output_pattern, text_data)
                                if output_match:
                                    output_text = output_match.group(1)
                                    logger.info(f"🤖 Extracted Agent workflow output via regex: {output_text}")
                                    return [{
                                        "result": output_text,
                                        "type": "agent_workflow_response", 
                                        "extraction_method": "regex",
                                        "timestamp": pd.Timestamp.now().isoformat()
                                    }]
                                
                                # Standard regex fallback for inventory data
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
            
            # Method 2: Enhanced string-based regex extraction
            response_text = str(mcp_response)
            import re
            
            # Look for Agent workflow output in the entire response
            output_pattern = r'"output":\s*"([^"]*)"'
            output_match = re.search(output_pattern, response_text)
            if output_match:
                output_text = output_match.group(1)
                logger.info(f"🤖 Found Agent workflow output in full response: {output_text}")
                return [{
                    "result": output_text,
                    "type": "agent_workflow_response",
                    "extraction_method": "full_response_regex",
                    "timestamp": pd.Timestamp.now().isoformat()
                }]
            
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
    
    def _format_data_with_claude(self, user_query: str, data: List[Dict], 
                               conversation_context: List[Dict] = None, tool_context: Dict = None) -> str:
        """Use Claude to format data like Claude Desktop"""
        try:
            # Check if this is Agent workflow response data
            is_agent_workflow = (
                len(data) > 0 and 
                isinstance(data[0], dict) and 
                data[0].get('type') == 'agent_workflow_response'
            )
            
            if is_agent_workflow:
                # Handle Agent workflow responses specially
                return self._format_agent_workflow_response(user_query, data, conversation_context)
            
            # Auto-detect if this should be formatted as table based on data structure
            should_use_table_format = self._should_auto_format_as_table(data, user_query)
            
            # Check if user specifically requested table format
            is_table_request = any(keyword in user_query.lower() for keyword in [
                'table', 'tabular', 'format', 'comprehensive', 'detailed analysis'
            ])
            
            # Use table format if auto-detected or specifically requested
            if should_use_table_format or is_table_request:
                logger.info("Auto-formatting structured data as table - showing ALL records")
                return self._create_comprehensive_table_analysis(data, tool_context or tool_info)
            
            # Prepare context for Claude
            context = self._prepare_formatting_context(user_query, data, conversation_context)
            
            # Get formatted response from Claude
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Same model as Claude Desktop
                max_tokens=8000,  # Higher token limit for large responses
                temperature=0.1,
                system="""You are a helpful assistant that formats BSL inventory data for users, working exactly like Claude Desktop. 
                        
Your job is to take raw inventory data and present it in a clear, human-readable format similar to how Claude Desktop would present it.

Key guidelines:
1. Always start with a brief summary of what was found
2. Present data in an organized, scannable format
3. Use clear headings and bullet points
4. Highlight important information like stock status, dates, and critical items
5. Be comprehensive but well-organized
6. Use markdown formatting for better readability
7. For large datasets, provide meaningful analysis and insights
8. Handle NCR data, PO data, and inventory data appropriately
9. Show key metrics and summaries

Make the output professional but easy to read at a glance. For large datasets, provide both summary and detailed views.""",
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude formatting error: {e}")
            # Fallback to basic formatting
            return self._basic_data_formatting(data)
    
    def _should_auto_format_as_table(self, data: List[Dict], user_query: str) -> bool:
        """
        Determine if structured data should automatically be formatted as table
        
        Args:
            data: The data to check
            user_query: Original user query
            
        Returns:
            True if should auto-format as table
        """
        if not data or not isinstance(data, list) or len(data) == 0:
            return False
        
        # Check if this is structured business data
        first_item = data[0] if data else {}
        
        # Business data indicators - these should automatically be tables
        business_data_indicators = [
            # DO/PO/Order data
            'POnumber', 'DOnumber', 'productCode', 'Ordered Date', 'Required Delievery Date',
            'Ordered Quantity', 'Current QOH', 'Stock Status',
            
            # NCR data
            'ncr', 'non_conformance', 'issue', 'severity', 'priority', 'NCMR',
            
            # Inventory data
            'inventory', 'stock', 'quantity', 'holding', 'location',
            
            # General business fields
            'row_number', 'ID', 'status', 'date', 'code'
        ]
        
        # Check if the data contains business data fields
        has_business_fields = any(
            any(indicator.lower() in str(key).lower() for indicator in business_data_indicators)
            for key in first_item.keys()
        ) if isinstance(first_item, dict) else False
        
        # Check query type - these queries should get table format
        table_query_indicators = [
            'check', 'list', 'show', 'get', 'find', 'display',
            'do', 'po', 'ncr', 'inventory', 'holding', 'treatment'
        ]
        
        query_suggests_table = any(
            indicator in user_query.lower() 
            for indicator in table_query_indicators
        )
        
        # Auto-format as table if:
        # 1. Has structured business data fields AND
        # 2. Query suggests listing/checking data
        should_format = has_business_fields and query_suggests_table
        
        if should_format:
            logger.info(f"Auto-detected table format needed for query: '{user_query}' with {len(data)} records")
        
        return should_format
    
    def _format_agent_workflow_response(self, user_query: str, data: List[Dict], 
                                       conversation_context: List[Dict] = None) -> str:
        """
        Format Agent workflow responses with Claude
        
        Args:
            user_query: The original user query
            data: Agent workflow response data
            conversation_context: Recent conversation context
            
        Returns:
            Formatted response from Claude
        """
        try:
            logger.info("🤖 Formatting Agent workflow response with Claude")
            
            # Extract the result from the data
            result_text = ""
            if len(data) > 0 and 'result' in data[0]:
                result_text = data[0]['result']
            
            # Prepare context for Claude formatting
            context = f"""User asked: "{user_query}"

Agent workflow returned this result:
{result_text}

Please format this response in a professional, clear, and user-friendly way similar to how Claude Desktop would present it.

Guidelines:
1. Make it conversational and natural
2. Use appropriate emojis and formatting
3. Provide context and explanation when helpful
4. For exchange rates, add currency symbols and source information
5. For approvals/rejections, use clear status indicators
6. For generation tasks, provide helpful next steps
7. Keep it concise but informative

Format the response as if you're directly answering the user's question with this information."""

            # Add conversation context if available
            if conversation_context:
                context += "\n\nRecent conversation context:\n"
                for msg in conversation_context[-2:]:  # Last 2 messages
                    role = "User" if msg.get('is_user') else "Assistant"
                    content = msg.get('content', '')[:100]  # Truncate
                    context += f"{role}: {content}\n"

            # Get formatted response from Claude
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1,
                system="""You are a helpful BSL inventory assistant formatting Agent workflow responses. 

Your job is to take the raw result from an Agent workflow tool and present it in a clear, natural way that directly answers the user's question.

Make the response:
- Conversational and natural (like Claude Desktop)
- Well-formatted with markdown
- Contextual and helpful
- Professional but friendly

Focus on directly answering what the user asked for.""",
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            
            formatted_response = response.content[0].text
            
            # Add metadata footer
            formatted_response += f"\n\n---\n*🤖 Processed by AI • {pd.Timestamp.now().strftime('%H:%M:%S')}*"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Agent workflow formatting error: {e}")
            # Fallback formatting
            result_text = data[0].get('result', 'Response received') if len(data) > 0 else 'Response received'
            return f"**Response:** {result_text}\n\n*🤖 Automated response from Agent workflow*"
    
    def _handle_large_dataset(self, user_query: str, data: List[Dict], 
                            conversation_context: List[Dict] = None) -> str:
        """Handle large datasets (like NCR lists) similar to Claude Desktop"""
        try:
            logger.info(f"Handling large dataset with {len(data)} items")
            
            # Create a comprehensive summary for large datasets
            df = pd.DataFrame(data)
            
            # Generate summary statistics
            summary_stats = {
                "total_items": len(df),
                "columns": list(df.columns),
                "column_count": len(df.columns)
            }
            
            # Analyze the data to provide meaningful insights
            insights = []
            
            # Common patterns for different types of data
            if any(col.lower() in ['ncr', 'non_conformance', 'issue'] for col in df.columns):
                insights.append("📋 **NCR Data Detected**")
                
                # NCR-specific analysis
                if 'Status' in df.columns:
                    status_counts = df['Status'].value_counts()
                    for status, count in status_counts.head(5).items():
                        insights.append(f"   - {status}: {count} items")
                
                if 'Priority' in df.columns or 'Severity' in df.columns:
                    priority_col = 'Priority' if 'Priority' in df.columns else 'Severity'
                    high_priority = df[df[priority_col].str.contains('High|Critical|Urgent', case=False, na=False)]
                    if len(high_priority) > 0:
                        insights.append(f"   - 🚨 High Priority Items: {len(high_priority)}")
                
                if 'Date' in df.columns or 'Created' in df.columns:
                    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                    if date_col:
                        try:
                            df[f'{date_col}_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                            recent_items = df[df[f'{date_col}_parsed'] > pd.Timestamp.now() - pd.Timedelta(days=30)]
                            insights.append(f"   - 📅 Recent (30 days): {len(recent_items)} items")
                        except:
                            pass
            
            # Show sample data (first few rows) with key columns
            sample_size = min(10, len(df))
            key_columns = df.columns[:8] if len(df.columns) > 8 else df.columns
            sample_df = df[key_columns].head(sample_size)
            
            # Create the formatted response using Claude for the sample
            sample_context = f"""
User asked: '{user_query}'

Dataset Summary:
- Total items: {summary_stats['total_items']}
- Columns: {', '.join(summary_stats['columns'][:10])}{'...' if len(summary_stats['columns']) > 10 else ''}

Key Insights:
{chr(10).join(insights) if insights else '- Large dataset detected'}

Sample Data (first {sample_size} rows):
{sample_df.to_json(orient='records', indent=2)}

Please format this as a comprehensive summary that shows:
1. Overview of the dataset
2. Key statistics and insights
3. Sample data in a readable format
4. Actionable next steps for the user

Make it clear this is a large dataset and offer ways to filter or analyze specific parts.
"""
            
            # Use Claude to format the summary
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                temperature=0.1,
                system="""You are formatting a large dataset summary for BSL inventory management. 
                
Provide a comprehensive yet organized view that helps users understand:
- What they're looking at
- Key metrics and patterns
- Sample of the actual data
- How to work with this large dataset

Use markdown formatting and make it scannable.""",
                messages=[
                    {"role": "user", "content": sample_context}
                ]
            )
            
            formatted_response = response.content[0].text
            
            # Add additional help for large datasets
            additional_help = f"""

---

## 🔧 **Working with Large Datasets**

Since this dataset has **{len(data)} items**, you can:

- **Filter by criteria**: "Show me NCRs with high priority"
- **Date range queries**: "Show me NCRs from last month"
- **Status filtering**: "Show me open NCRs only"
- **Search specific items**: "Find NCR containing [keyword]"
- **Export/Table format**: "Show this in table format"

*💡 The complete dataset is available - just ask for specific filters or analysis!*
"""
            
            return formatted_response + additional_help
            
        except Exception as e:
            logger.error(f"Large dataset handling error: {e}")
            return f"""
## 📊 Large Dataset Summary

**Total Items Found:** {len(data)}

This is a large dataset with {len(data)} items. Due to the size, here's a summary:

**Sample Data (first 5 items):**
{json.dumps(data[:5], indent=2)}

**Dataset Info:**
- Total records: {len(data)}
- Available columns: {list(data[0].keys()) if data else 'Unknown'}

**Next Steps:**
Ask for specific filters or analysis like:
- "Show me items with status X"
- "Filter by date range"
- "Show in table format"
- "Analyze by category"

*💡 The full dataset is available - just specify what you'd like to see!*
"""
    
    def _prepare_formatting_context(self, user_query: str, data: List[Dict], 
                                  conversation_context: List[Dict] = None) -> str:
        """Prepare context for Claude formatting"""
        
        context_parts = [
            f"User asked: '{user_query}'",
            "",
            f"Raw data retrieved ({len(data)} items):",
        ]
        
        # For large datasets, show sample + summary instead of full data
        if len(data) > 20:
            sample_data = data[:15]  # Show first 15 items for context
            context_parts.extend([
                f"Sample data (first 15 of {len(data)} items):",
                json.dumps(sample_data, indent=2),
                f"... and {len(data) - 15} more items (full dataset available for analysis)",
                "",
                "This is a large dataset. Provide a comprehensive summary with key insights."
            ])
        else:
            context_parts.extend([
            json.dumps(data, indent=2),
                ""
            ])
        
        context_parts.extend([
            "",
            "Please format this data in a clear, human-readable way that directly answers the user's question.",
            "Focus on the most important information and present it professionally."
        ])
        
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
        
        # If this is still showing raw JSON structure, let's try to force Claude formatting
        if '"content"' in raw_text and '"text"' in raw_text:
            logger.info("Detected raw JSON structure, attempting forced Claude formatting...")
            
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
                        # Force Claude formatting as last resort
                        logger.info(f"Forcing Claude formatting for {len(data)} items...")
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
    
    def process_table_format_request(self, user_query: str, data: List[Dict], 
                                   conversation_context: List[Dict] = None, tool_context: Dict = None) -> str:
        """Handle table format requests with comprehensive Claude Desktop-style analysis"""
        
        logger.info(f"Processing table format request for {len(data)} items")
        
        # Always use comprehensive analysis for table format requests
        return self._create_comprehensive_table_analysis(data, tool_context)
    
    def _extract_nested_results(self, data: List[Dict]) -> List[Dict]:
        """Extract nested JSON from 'result' fields in data items"""
        try:
            if not data or len(data) == 0:
                return data
                
            # Check if first item has a "result" field with JSON string
            if (isinstance(data[0], dict) and 
                'result' in data[0] and 
                isinstance(data[0]['result'], str)):
                
                logger.info("🔄 Extracting nested JSON from 'result' fields...")
                extracted_records = []
                
                for item in data:
                    if 'result' in item and isinstance(item['result'], str):
                        try:
                            # Parse the JSON string inside the result field
                            nested_record = json.loads(item['result'])
                            extracted_records.append(nested_record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse nested result JSON: {e}")
                            # Keep the original item if parsing fails
                            extracted_records.append(item)
                    else:
                        extracted_records.append(item)
                
                logger.info(f"✅ Successfully extracted {len(extracted_records)} records from nested results")
                return extracted_records
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting nested results: {e}")
            return data 
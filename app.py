import streamlit as st
import time
import os
import base64
from datetime import datetime
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pytz  # Add this import
import json # Added for json.loads

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from config import get_app_config, get_chat_config, get_mcp_server_url, get_health_check_url
from mcp_client import MCPClient

def get_base64_image(image_path):
    """Convert image to base64 string for embedding in HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Configure the page
app_config = get_app_config()
st.set_page_config(**app_config)

# Set timezone for Singapore
SGT = pytz.timezone('Asia/Singapore')

def get_current_time():
    """Get current time in Singapore timezone"""
    return datetime.now(SGT)

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp to Singapore time"""
    try:
        # Parse the timestamp and convert to Singapore time
        if timestamp_str.endswith('Z'):
            # If UTC timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            # If ISO format without timezone
            dt = datetime.fromisoformat(timestamp_str)
            if dt.tzinfo is None:
                # If no timezone info, assume UTC
                dt = dt.replace(tzinfo=pytz.UTC)
        
        # Convert to Singapore time
        sgt_time = dt.astimezone(SGT)
        return sgt_time.strftime('%H:%M:%S')
    except Exception:
        # Fallback to current time if parsing fails
        return get_current_time().strftime('%H:%M:%S')

def initialize_session_state():
    """Initialize session state variables"""
    chat_config = get_chat_config()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "mcp_client" not in st.session_state:
        # Initialize with Claude API key from environment variable or Streamlit secrets
        claude_key = os.getenv('ANTHROPIC_API_KEY')
        if not claude_key:
            try:
                # Try Streamlit secrets (for Streamlit Cloud deployment)
                claude_key = st.secrets["ANTHROPIC_API_KEY"]
            except (KeyError, FileNotFoundError):
                st.warning("⚠️ Claude API key not found. Set ANTHROPIC_API_KEY environment variable or add it to Streamlit secrets for AI features.")
        st.session_state.mcp_client = MCPClient(claude_api_key=claude_key)
    if "chat_config" not in st.session_state:
        st.session_state.chat_config = chat_config

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message in the UI"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.write(message["content"])
        else:
            if message.get("success", True):
                content = message["content"]
                
                # Check if this is an error response in content
                if isinstance(content, dict) and content.get('type') == 'error_response':
                    st.error(content.get('error', 'An error occurred'))
                    return
                
                # Check if this is an AI-generated response with special formatting
                source = message.get("source", "")
                tool_used = message.get("tool_used", "")
                tool_type = message.get("tool_type", "")
                
                if source in ["ai_processor", "ai_table_formatter", "claude_formatted", "agent_workflow_fallback"]:
                    # Display AI responses as plain text only - no formatting at all
                    import re
                    # Strip ALL markdown formatting for consistent font
                    plain_content = content
                    plain_content = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_content)  # Remove bold
                    plain_content = re.sub(r'\*(.*?)\*', r'\1', plain_content)  # Remove italic  
                    plain_content = re.sub(r'#{1,6}\s*', '', plain_content)  # Remove headers
                    plain_content = re.sub(r'`(.*?)`', r'\1', plain_content)  # Remove code formatting
                    plain_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', plain_content)  # Remove links
                    plain_content = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', plain_content)  # Remove underline
                    plain_content = re.sub(r'~~(.*?)~~', r'\1', plain_content)  # Remove strikethrough
                    # Use st.text for completely plain text display
                    st.text(plain_content)
                    
                    # Show appropriate status indicators
                    if source == "claude_formatted":
                        if tool_type == "agent_workflow":
                            st.success("🤖 Formatted by Claude • Agent Workflow")
                        elif tool_type == "record_agent":
                            st.success("🤖 Formatted by Claude • Record Agent")
                        else:
                            st.success("🤖 Formatted by Claude")
                    elif source == "agent_workflow_fallback":
                        st.info("⚡ Agent Workflow • Fallback Formatting")
                    elif source == "ai_table_formatter":
                        st.success("📊 Table formatted by AI")
                    elif source == "ai_processor":
                        st.info("🧠 Processed by AI")
                        
                    # Show tool information if available
                    if tool_used and tool_used != "unknown":
                        st.caption(f"Tool: {tool_used}")
                        
                else:
                    # Regular MCP response - try to format JSON nicely
                    try:
                        # Check if this is a raw JSON error message
                        if isinstance(content, str):
                            try:
                                data = json.loads(content)
                                if isinstance(data, dict) and ('error' in data or 'message' in data):
                                    error_msg = data.get('error') or data.get('message')
                                    st.error(f"🚫 {error_msg}")
                                    return
                                
                                # Handle empty array responses
                                if isinstance(data, dict) and 'content' in data:
                                    content_list = data['content']
                                    if isinstance(content_list, list) and len(content_list) > 0:
                                        first_item = content_list[0]
                                        if isinstance(first_item, dict) and 'text' in first_item:
                                            text = first_item['text']
                                            if text == "[]" or text == '[{"type":"text","text":"[]"}]':
                                                st.info("📭 No data available at the moment.")
                                                return
                                
                                # Handle direct empty array
                                if isinstance(data, list) and len(data) == 0:
                                    st.info("📭 No data available at the moment.")
                                    return
                                    
                            except:
                                pass
                        
                        # Check for empty array in plain text
                        if isinstance(content, str):
                            if content.strip() == "[]" or content.strip() == '[{"type":"text","text":"[]"}]':
                                st.info("📭 No data available at the moment.")
                                return
                        
                        if str(content).startswith('[') and len(str(content)) > 100:
                            st.info("📊 Raw data received - try asking for 'table format' for better display")
                        elif str(content).startswith('{') and 'content' in str(content):
                            st.warning("🔧 Raw MCP response - enable AI formatting for better display")
                        
                        # Apply same plain text treatment to all responses for consistent font
                        import re
                        plain_content = str(content)
                        plain_content = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_content)  # Remove bold
                        plain_content = re.sub(r'\*(.*?)\*', r'\1', plain_content)  # Remove italic  
                        plain_content = re.sub(r'#{1,6}\s*', '', plain_content)  # Remove headers
                        plain_content = re.sub(r'`(.*?)`', r'\1', plain_content)  # Remove code formatting
                        plain_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', plain_content)  # Remove links
                        plain_content = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', plain_content)  # Remove underline
                        plain_content = re.sub(r'~~(.*?)~~', r'\1', plain_content)  # Remove strikethrough
                        st.text(plain_content)
                    except:
                        # Apply plain text treatment to fallback content too
                        import re
                        plain_content = str(content)
                        plain_content = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_content)  # Remove bold
                        plain_content = re.sub(r'\*(.*?)\*', r'\1', plain_content)  # Remove italic  
                        plain_content = re.sub(r'#{1,6}\s*', '', plain_content)  # Remove headers
                        plain_content = re.sub(r'`(.*?)`', r'\1', plain_content)  # Remove code formatting
                        plain_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', plain_content)  # Remove links
                        plain_content = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', plain_content)  # Remove underline
                        plain_content = re.sub(r'~~(.*?)~~', r'\1', plain_content)  # Remove strikethrough
                        st.text(plain_content)
            else:
                # Get the error message
                error_msg = message.get('error', 'Unknown error')
                
                # Clean up common error patterns
                if "Input was rejected for safety reasons" in error_msg:
                    st.error("⚠️ Please refer to 'Available Commands' and try again.")
                elif "Could not establish session" in error_msg:
                    st.error("🔌 Connection issue - please try again in a moment.")
                elif "Failed to get data from MCP server" in error_msg:
                    st.error("🔄 Server communication error - please try again.")
                elif "Timeout" in error_msg:
                    st.error("⏳ Request took too long - please try again.")
                elif "Empty response" in error_msg:
                    st.error("📭 No data received - please try a different query.")
                else:
                    # For unknown errors, show a user-friendly message
                    st.error(f"🚫 {error_msg}")
        
        # Add timestamp with source info
        timestamp = message.get("timestamp", get_current_time().isoformat())
        source_info = f" • {message.get('source', 'mcp_server')}" if not is_user else ""
        st.caption(f"*{format_timestamp(timestamp)}{source_info}*")

def main():
    initialize_session_state()
    
    # Simple compact title
    st.title("🤖 BSL AI Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        # Add logos at the top of sidebar
        st.markdown("""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 20px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        ">
            <div style="
                background: rgba(255,255,255,0.9);
                padding: 8px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <img src="data:image/png;base64,{}" style="height: 40px; width: auto;" />
            </div>
            <span style="font-size: 20px; color: #666; font-weight: 600;">×</span>
            <div style="
                background: rgba(255,255,255,0.9);
                padding: 8px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <img src="data:image/jpeg;base64,{}" style="height: 40px; width: auto;" />
            </div>
        </div>
        """.format(
            get_base64_image("image/3echo.png"),
            get_base64_image("image/bsl.jpg")
        ), unsafe_allow_html=True)
        
        # Available commands
        st.subheader("Available Commands")
        commands = st.session_state.mcp_client.get_available_commands()
        with st.expander("📋 Show Commands", expanded=False):
            # Display commands by category with enhanced structure
            for category, category_commands in commands.items():
                st.markdown(f"**{category}**")
                
                if isinstance(category_commands, dict):
                    for command, description in category_commands.items():
                        st.markdown(f"  • **{command}**: {description}")
                else:
                    st.markdown(f"  {category_commands}")
                
                st.markdown("")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Set default values for removed configuration options
    auto_scroll = st.session_state.chat_config.get("auto_scroll", True)
    stream_responses = True
    max_messages = st.session_state.chat_config.get("max_messages", 50)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        # Limit messages displayed based on max_messages setting
        messages_to_show = st.session_state.messages[-max_messages:] if len(st.session_state.messages) > max_messages else st.session_state.messages
        
        for i, message in enumerate(messages_to_show):
            display_chat_message(message["user_message"], is_user=True)
            display_chat_message(message["assistant_response"], is_user=False)
            
            # Add separator between conversations
            if i < len(messages_to_show) - 1:
                st.markdown("---")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        user_message = {
            "content": user_input,
            "timestamp": get_current_time().isoformat()
        }
        
        # Display user message immediately
        with chat_container:
            display_chat_message(user_message, is_user=True)
        
        # Show spinner while processing
        with st.spinner("Processing your request..."):
            # Prepare conversation context for AI
            conversation_context = []
            for msg_pair in st.session_state.messages[-3:]:  # Last 3 exchanges
                conversation_context.extend([
                    {"content": msg_pair["user_message"]["content"], "is_user": True},
                    {"content": msg_pair["assistant_response"].get("content", ""), "is_user": False}
                ])
            
            # Send to MCP server with AI capabilities
            response = st.session_state.mcp_client.send_query(
                user_input, 
                st.session_state.conversation_id,
                stream=stream_responses,
                conversation_context=conversation_context
            )
            
            # Prepare assistant response
            if response["success"]:
                assistant_message = {
                    "content": response["response"],
                    "timestamp": response["timestamp"],
                    "success": True
                }
            else:
                assistant_message = {
                    "content": "",
                    "error": response["error"],
                    "timestamp": get_current_time().isoformat(),
                    "success": False
                }
        
        # Add to conversation history
        st.session_state.messages.append({
            "user_message": user_message,
            "assistant_response": assistant_message
        })
        
        # Display assistant response
        with chat_container:
            display_chat_message(assistant_message, is_user=False)
        
        # Auto-scroll if enabled
        if auto_scroll:
            st.rerun()

if __name__ == "__main__":
    main() 
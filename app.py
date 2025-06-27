import streamlit as st
import time
import os
from datetime import datetime
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from config import get_app_config, get_chat_config, get_mcp_server_url, get_health_check_url
from mcp_client import MCPClient

# Configure the page
app_config = get_app_config()
st.set_page_config(**app_config)

def initialize_session_state():
    """Initialize session state variables"""
    chat_config = get_chat_config()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "mcp_client" not in st.session_state:
        # Initialize with OpenAI API key from environment variable or Streamlit secrets
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            try:
                # Try Streamlit secrets (for Streamlit Cloud deployment)
                openai_key = st.secrets["OPENAI_API_KEY"]
            except (KeyError, FileNotFoundError):
                st.warning("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable or add it to Streamlit secrets for AI features.")
        st.session_state.mcp_client = MCPClient(openai_api_key=openai_key)
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
                
                # Check if this is an AI-generated response with special formatting
                source = message.get("source", "")
                if source in ["ai_processor", "ai_table_formatter"]:
                    # Display AI responses with markdown support
                    st.markdown(content)
                    if source == "ai_table_formatter":
                        st.success("🤖 Formatted by AI")
                    elif source == "ai_processor":
                        st.info("🧠 Processed by AI")
                else:
                    # Regular MCP response - try to format JSON nicely
                    try:
                        if content.startswith('[') and len(content) > 100:
                            st.info("📊 Raw data received - try asking for 'table format' for better display")
                        st.write(content)
                    except:
                        st.write(content)
            else:
                st.error(f"Error: {message.get('error', 'Unknown error')}")
        
        # Add timestamp with source info
        timestamp = message.get("timestamp", datetime.now().isoformat())
        source_info = f" • {message.get('source', 'mcp_server')}" if not is_user else ""
        st.caption(f"*{datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%H:%M:%S')}{source_info}*")

def main():
    initialize_session_state()
    
    # Header
    st.title("🤖 BSL AI Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Server status
        st.subheader("Server Status")
        health_status = st.session_state.mcp_client.health_check()
        
        if health_status["status"] == "healthy":
            st.success(f"🟢 {health_status['message']}")
            if "response_time" in health_status:
                st.caption(f"Response time: {health_status['response_time']:.3f}s")
        elif health_status["status"] == "timeout":
            st.warning(f"🟡 {health_status['message']}")
        else:
            st.error(f"🔴 {health_status['message']}")
        
        # AI Status
        if hasattr(st.session_state.mcp_client, 'ai_processor') and st.session_state.mcp_client.ai_processor:
            st.success("🤖 AI Enhanced Mode Active")
            st.caption("Can handle complex queries, data formatting, and analysis")
        else:
            st.warning("🔧 Basic Mode Only")
            st.caption("Direct tool calls only")
        
        st.info(f"**Server URL:** {get_mcp_server_url()}")
        
        # Available commands
        st.subheader("Available Commands")
        commands = st.session_state.mcp_client.get_available_commands()
        with st.expander("📋 Show Commands", expanded=False):
            for cmd, desc in commands.items():
                st.write(f"**{cmd}**")
                st.caption(desc)
                st.write("")
        
        # Debug: Server Tools Discovery
        st.subheader("🔍 Debug Tools")
        if st.button("Discover Server Tools"):
            with st.spinner("Querying server for available tools..."):
                tools_info = st.session_state.mcp_client.get_available_tools()
                
                if tools_info["success"]:
                    if "tools" in tools_info:
                        st.success("✅ Found server tools!")
                        st.json(tools_info["tools"])
                    else:
                        st.info("Server response:")
                        st.text(tools_info.get("response", "No response"))
                else:
                    st.error(f"❌ Failed to get tools: {tools_info['error']}")
        
        # Conversation controls
        st.subheader("Conversation")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
        
        st.info(f"**Conversation ID:** {st.session_state.conversation_id[:8]}...")
        
        # Settings
        st.subheader("Settings")
        auto_scroll = st.checkbox("Auto-scroll to bottom", 
                                 value=st.session_state.chat_config["auto_scroll"])
        show_timestamps = st.checkbox("Show timestamps", 
                                     value=st.session_state.chat_config["show_timestamps"])
        stream_responses = st.checkbox("Stream responses", value=True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            response_timeout = st.slider("Response timeout (seconds)", 
                                       min_value=10, max_value=60, 
                                       value=st.session_state.chat_config["response_timeout"])
            max_messages = st.slider("Max messages in history", 
                                   min_value=10, max_value=200, 
                                   value=st.session_state.chat_config["max_messages"])
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
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
            "timestamp": datetime.now().isoformat()
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
                    "timestamp": datetime.now().isoformat(),
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
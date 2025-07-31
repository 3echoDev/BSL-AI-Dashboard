"""Configuration settings for BSL AI Dashboard"""

import os
from typing import Dict, Any

# MCP Server Configuration
MCP_SERVER_CONFIG = {
    "base_url": "https://bslunifyone.app.n8n.cloud/mcp/ClaudeDesktopMCP",
    "sse_endpoint": "/sse",
    "messages_endpoint": "/messages",
    "health_endpoint": "/health",
    "timeout": 30,
    "max_retries": 3
}

# Streamlit App Configuration
APP_CONFIG = {
    "page_title": "BSL AI Dashboard",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Chat Configuration
CHAT_CONFIG = {
    "max_messages": 100,  # Maximum number of messages to keep in session
    "auto_scroll": True,
    "show_timestamps": True,
    "response_timeout": 30
}

# UI Theme Configuration
UI_CONFIG = {
    "primary_color": "#FF6B6B",
    "background_color": "#FFFFFF",
    "secondary_background_color": "#F0F2F6",
    "text_color": "#262730"
}

def get_mcp_server_url() -> str:
    """Get the complete MCP server SSE URL"""
    return f"{MCP_SERVER_CONFIG['base_url']}{MCP_SERVER_CONFIG['sse_endpoint']}"

def get_health_check_url() -> str:
    """Get the health check URL"""
    return f"{MCP_SERVER_CONFIG['base_url']}{MCP_SERVER_CONFIG['health_endpoint']}"

def get_app_config() -> Dict[str, Any]:
    """Get Streamlit app configuration"""
    return APP_CONFIG

def get_chat_config() -> Dict[str, Any]:
    """Get chat configuration"""
    return CHAT_CONFIG 
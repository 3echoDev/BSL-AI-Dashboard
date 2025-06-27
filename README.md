# BSL AI Dashboard

A modern Streamlit-based chat interface for interacting with MCP (Model Context Protocol) servers. This dashboard provides a clean, user-friendly interface to query your n8n-hosted MCP server.

## 🚀 Features

- **Real-time Chat Interface**: Interactive chat UI with message history
- **Server-Sent Events (SSE) Support**: Real-time streaming responses from MCP server
- **Health Monitoring**: Live server status monitoring with response time tracking
- **Conversation Management**: Persistent conversation sessions with unique IDs
- **Customizable Settings**: Adjustable timeouts, message limits, and UI preferences
- **Error Handling**: Robust error handling with retry mechanisms
- **Responsive Design**: Modern UI that works on desktop and mobile devices

## 📋 Prerequisites

- Python 3.8 or higher
- Access to your n8n MCP server
- Internet connection for API calls

## 🛠️ Installation

1. **Clone or download the project files**:
   ```bash
   # If using git
   git clone https://github.com/shinkaung/BSL-AI-Dashboard.git
   cd BSL-AI-Dashboard
   
   # Or download and extract the files to a directory
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your-actual-api-key-here
   ```

## ⚙️ Configuration

The application is pre-configured to connect to your n8n MCP server at:
```
https://threeecho.app.n8n.cloud/mcp/ClaudeDesktopMCP/sse
```

If you need to modify the server URL or other settings, edit the `config.py` file:

```python
# MCP Server Configuration
MCP_SERVER_CONFIG = {
    "base_url": "https://your-server.com/mcp/endpoint",
    "sse_endpoint": "/sse",
    "health_endpoint": "/health",
    "timeout": 30,
    "max_retries": 3
}
```

## 🚀 Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Start chatting**:
   - Type your message in the chat input at the bottom
   - Press Enter or click Send
   - Watch for real-time responses from your MCP server

## 🎛️ Dashboard Features

### Sidebar Controls

- **Server Status**: Real-time health monitoring of your MCP server
- **Conversation Management**: Clear chat history and start new conversations
- **Settings**: 
  - Auto-scroll to bottom
  - Show/hide timestamps
  - Enable/disable streaming responses
- **Advanced Settings**:
  - Response timeout configuration
  - Maximum message history limit

### Main Chat Interface

- **Message History**: Scrollable conversation history
- **Real-time Responses**: Streaming responses from the server
- **Error Display**: Clear error messages when something goes wrong
- **Timestamps**: Optional message timestamps

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**:
   - Check if your n8n server is running
   - Verify the server URL in `config.py`
   - Check your internet connection

2. **Timeout Errors**:
   - Increase the timeout value in settings
   - Check server response time in the sidebar

3. **Empty Responses**:
   - Verify your MCP server is properly configured
   - Check server logs for errors

4. **Installation Issues**:
   ```bash
   # Update pip first
   pip install --upgrade pip
   
   # Install dependencies one by one if bulk install fails
   pip install streamlit
   pip install requests
   pip install sseclient-py
   pip install python-dateutil
   ```

### Debug Mode

To enable debug logging, modify the logging level in `mcp_client.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📁 Project Structure

```
BSL AI/
├── app.py              # Main Streamlit application
├── config.py           # Configuration settings
├── mcp_client.py       # MCP server client implementation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔒 Security Notes

- The application connects to external servers (your n8n instance)
- Ensure your n8n server has proper authentication if needed
- Consider using HTTPS for production deployments
- Review firewall settings if having connection issues

## 🤝 Support

If you encounter issues:

1. Check the server status in the sidebar
2. Review the error messages in the chat interface
3. Check the console output where you ran `streamlit run app.py`
4. Verify your n8n MCP server configuration

## 🌐 Deployment

### Local Development
See installation instructions above.

### Streamlit Cloud Deployment

1. **Push to GitHub** (the .env file will NOT be included):
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `shinkaung/BSL-AI-Dashboard`
   - Branch: `main`
   - Main file: `app.py`

3. **Configure Environment Variables**:
   In the "Advanced settings" before deploying, add your secrets:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

4. **Deploy**: Click "Deploy!" and your app will be live at a Streamlit Cloud URL.

### Other Deployment Options
- **Heroku**: Use `heroku config:set OPENAI_API_KEY=your-key`
- **Railway**: Add environment variables in the dashboard
- **Docker**: Pass environment variables with `-e OPENAI_API_KEY=your-key`

## 🔄 Updates

To update the application:
1. Update the Python files with any changes
2. Restart the Streamlit application
3. Refresh your browser

---

**Built with ❤️ using Streamlit and Python** 
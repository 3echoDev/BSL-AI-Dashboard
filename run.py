#!/usr/bin/env python3
"""
BSL AI Dashboard Launcher
Simple script to start the Streamlit application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import requests
        import sseclient
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main launcher function"""
    print("🤖 BSL AI Dashboard Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🚀 Starting Streamlit application...")
    print("📱 The dashboard will open in your default browser")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
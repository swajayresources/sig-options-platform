"""
Application Runner
Launch the Streamlit options trading platform with proper configuration
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
 """Check if all required packages are installed"""
 required_packages = [
 'streamlit',
 'plotly',
 'pandas',
 'numpy',
 'scipy',
 'yfinance'
 ]

 missing_packages = []

 for package in required_packages:
 try:
 __import__(package)
 except ImportError:
 missing_packages.append(package)

 if missing_packages:
 print("❌ Missing required packages:")
 for package in missing_packages:
 print(f" - {package}")
 print("\n📦 Install missing packages with:")
 print(f" pip install {' '.join(missing_packages)}")
 return False

 print("✅ All required packages are installed")
 return True

def setup_environment():
 """Setup the environment for the application"""
 # Create necessary directories
 os.makedirs('.streamlit', exist_ok=True)
 os.makedirs('data', exist_ok=True)
 os.makedirs('logs', exist_ok=True)

 print("📁 Directory structure created")

def run_streamlit_app():
 """Run the Streamlit application"""
 print("🚀 Starting Professional Options Trading Platform...")
 print("📊 Platform will be available at: http://localhost:8501")
 print("\n" + "="*60)
 print("PROFESSIONAL OPTIONS TRADING PLATFORM")
 print("="*60)
 print("Features:")
 print("• Real-time portfolio monitoring")
 print("• Interactive options chain analysis")
 print("• 3D volatility surface visualization")
 print("• Advanced risk management")
 print("• Options flow and sentiment analysis")
 print("• Strategy performance tracking")
 print("="*60)

 # Run the Streamlit app
 subprocess.run([
 sys.executable, '-m', 'streamlit', 'run', 'main.py',
 '--server.port=8501',
 '--server.address=0.0.0.0',
 '--browser.gatherUsageStats=false'
 ])

def main():
 """Main application entry point"""
 print("🎯 Professional Options Trading Platform")
 print(" Streamlit-based institutional trading interface")
 print()

 # Check requirements
 if not check_requirements():
 sys.exit(1)

 # Setup environment
 setup_environment()

 # Run the application
 try:
 run_streamlit_app()
 except KeyboardInterrupt:
 print("\n\n👋 Application stopped by user")
 except Exception as e:
 print(f"\n❌ Error starting application: {e}")
 sys.exit(1)

if __name__ == "__main__":
 main()
"""
Script to test connection to the local LLM server.
"""

import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_llm_connection():
    """Test connection to the local LLM server."""
    print("🔄 Testing LLM connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get LLM configuration
    api_base = os.getenv("LLM_API_BASE", "http://localhost:8000")
    
    try:
        # Test server connection
        response = requests.get(f"{api_base}/health")
        if response.status_code == 200:
            print("✅ LLM server is running and accessible")
            return True
        else:
            print(f"❌ LLM server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to LLM server")
        print(f"   Make sure your LLM server is running at: {api_base}")
        print("\nTo start a local LLM server:")
        print("1. Install a local LLM (e.g., Mistral-7B)")
        print("2. Set up a server using tools like vLLM or FastAPI")
        print("3. Update LLM_API_BASE in .env if using a different address")
        return False
    except Exception as e:
        print(f"❌ Error testing LLM connection: {e}")
        return False

def test_llm_completion():
    """Test LLM completion functionality."""
    print("\n🔄 Testing LLM completion...")
    
    # Load environment variables
    load_dotenv()
    
    # Get LLM configuration
    api_base = os.getenv("LLM_API_BASE", "http://localhost:8000")
    
    try:
        # Test completion endpoint
        response = requests.post(
            f"{api_base}/v1/completions",
            json={
                "model": "mistral-7b",
                "prompt": "Hello, how are you?",
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                print("✅ LLM completion test successful")
                print(f"Response: {result['choices'][0]['text']}")
                return True
        
        print(f"❌ LLM completion test failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False
        
    except Exception as e:
        print(f"❌ Error testing LLM completion: {e}")
        return False

if __name__ == "__main__":
    print("🔍 LLM Connection Test")
    print("=====================")
    
    connection_success = test_llm_connection()
    if connection_success:
        test_llm_completion()
    
    print("\nℹ️ For help setting up your local LLM, check the README.md file.")

"""
Complete setup script for the Social Media Manager system.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_step(step: str):
    """Print a step header."""
    print(f"\n{'='*80}")
    print(f"Step: {step}")
    print(f"{'='*80}\n")

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def install_pytorch() -> bool:
    """Install PyTorch (CPU version)."""
    print_step("Installing PyTorch")
    
    # Install CPU-only version
    return run_command(
        "pip install torch torchvision torchaudio",
        "PyTorch installation (CPU)"
    )

def install_dependencies(activate_cmd: str) -> bool:
    """Install dependencies in the correct order."""
    print_step("Installing dependencies")
    
    # First, upgrade pip
    if not run_command(
        f"{activate_cmd} && python -m pip install --upgrade pip",
        "Pip upgrade"
    ):
        return False
    
    # Install base dependencies first
    if not run_command(
        f"{activate_cmd} && pip install python-dotenv==1.0.0 requests==2.31.0 aiohttp==3.9.1 python-dateutil==2.8.2 rich==13.7.0",
        "Base dependencies installation"
    ):
        return False
    
    # Install PyTorch
    if not install_pytorch():
        print("\n❌ PyTorch installation failed")
        return False
    
    # Install AI dependencies
    if not run_command(
        f"{activate_cmd} && pip install transformers>=4.36.0 huggingface-hub==0.20.3 pyautogen==0.7.5",
        "AI dependencies installation"
    ):
        return False
    
    # Install web dependencies
    if not run_command(
        f"{activate_cmd} && pip install fastapi==0.109.2 uvicorn==0.27.1 pydantic==2.6.1",
        "Web dependencies installation"
    ):
        return False
    
    # Install test dependencies
    if not run_command(
        f"{activate_cmd} && pip install pytest==7.4.3 pytest-asyncio==0.23.2",
        "Test dependencies installation"
    ):
        return False
    
    return True

def main():
    """Main setup function."""
    print("\n🚀 Social Media Manager Setup")
    print("============================")
    
    # Create virtual environment
    print_step("Creating virtual environment")
    if not run_command("python -m venv venv", "Virtual environment creation"):
        return
    
    # Activate virtual environment
    print_step("Activating virtual environment")
    if os.name == "nt":  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/MacOS
        activate_cmd = "source venv/bin/activate"
    
    if not run_command(f"{activate_cmd} && python -c \"print('Virtual environment activated')\"",
                      "Virtual environment activation"):
        return
    
    # Install dependencies
    if not install_dependencies(activate_cmd):
        return
    
    # Create necessary directories
    print_step("Creating project directories")
    directories = ["logs", "cache", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    # Set up LLM server
    print_step("Setting up LLM server")
    if not run_command(f"{activate_cmd} && python setup_llm_server.py",
                      "LLM server setup"):
        return
    
    # Set up social media credentials
    print_step("Setting up social media credentials")
    if not run_command(f"{activate_cmd} && python setup_credentials.py",
                      "Social media credentials setup"):
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the LLM server:")
    print("   python run_llm_server.py")
    print("\n2. Test the setup:")
    print("   python test_llm_connection.py")
    print("\n3. Try the basic workflow:")
    print("   python examples/basic_workflow.py")
    
    print("\nOptional GPU Support:")
    print("If you have a CUDA-capable GPU, you can install GPU support with:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n📚 For more information, check the README.md file")

if __name__ == "__main__":
    main()

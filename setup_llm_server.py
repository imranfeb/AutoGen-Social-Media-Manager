"""
Script to help set up and run a local LLM server.
"""

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

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

def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ Found {device_count} CUDA GPU(s)")
            for i in range(device_count):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️ No CUDA GPU found - using CPU mode")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed - using CPU mode")
        return False

def install_llm_dependencies():
    """Install LLM server dependencies."""
    print_step("Installing LLM server dependencies")
    
    # Install FastAPI and Uvicorn
    if not run_command(
        "pip install fastapi==0.109.2 uvicorn==0.27.1",
        "Web server dependencies installation"
    ):
        return False
    
    # Install Hugging Face dependencies
    if not run_command(
        "pip install transformers>=4.36.0 accelerate>=0.27.0 safetensors>=0.4.0",
        "Hugging Face dependencies installation"
    ):
        return False
    
    return True

def download_model():
    """Download Mistral-7B model."""
    print_step("Downloading Mistral-7B model")
    try:
        from huggingface_hub import snapshot_download
        
        model_path = Path("models/mistral-7b")
        if not model_path.exists():
            print("Downloading model (this may take a while)...")
            snapshot_download(
                repo_id="mistralai/Mistral-7B-v0.1",
                local_dir=str(model_path),
                ignore_patterns=["*.md", "*.txt"]
            )
            print("✅ Model downloaded successfully")
        else:
            print("✅ Model already downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def create_server_script():
    """Create a script to run the LLM server."""
    script_content = '''"""
Script to run the local LLM server.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import uvicorn

app = FastAPI(title="Local LLM Server")

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading model and tokenizer...")
model_path = "models/mistral-7b"

# Configure model loading based on device
model_kwargs = {
    "device_map": "auto" if device == "cuda" else None,
    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    "low_cpu_mem_usage": True if device == "cpu" else False
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1
)

print("Model loaded successfully!")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stop: Optional[List[str]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": device,
        "model": model_path
    }

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    try:
        # Generate text
        result = generator(
            request.prompt,
            max_length=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )[0]
        
        return {
            "id": "cmpl-local",
            "object": "text_completion",
            "created": None,
            "model": model_path,
            "choices": [{
                "text": result["generated_text"],
                "index": 0,
                "finish_reason": "length"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    print(f"Starting LLM server with model: {model_path}")
    print(f"Device: {device}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
'''
    
    with open("run_llm_server.py", "w") as f:
        f.write(script_content)
    print("\n✅ Created server script: run_llm_server.py")

def main():
    """Main setup function."""
    print("🚀 Local LLM Server Setup")
    print("========================")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Install dependencies
    if not install_llm_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return
    
    # Download model
    if not download_model():
        print("\n❌ Setup failed: Could not download model")
        return
    
    # Create server script
    create_server_script()
    
    print("\n✅ Setup completed successfully!")
    print("\nTo start the LLM server:")
    print("1. Run: python run_llm_server.py")
    print("2. Test connection: python test_llm_connection.py")
    
    if not has_gpu:
        print("\n⚠️ Note: Running in CPU mode will be significantly slower.")
        print("To enable GPU support, install CUDA and PyTorch with:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nThe server will be available at: http://localhost:8000")

if __name__ == "__main__":
    main()

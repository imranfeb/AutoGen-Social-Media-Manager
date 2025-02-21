# Social Media Manager

A multi-agent system for automated social media management using AutoGen framework.

## Features
- Multi-platform content management (Twitter, Facebook, LinkedIn, Instagram)
- AI-powered content generation using local LLM
- Automated scheduling and posting
- Analytics and performance tracking
- Platform-specific content optimization

## System Requirements
- Python 3.11 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space for LLM model

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the local LLM server:
```bash
python setup_llm_server.py
```
This script will:
- Check for GPU availability
- Install vLLM and dependencies
- Download the Mistral-7B model
- Create the server script

4. Start the LLM server:
```bash
python run_llm_server.py
```

5. Test the LLM connection:
```bash
python test_llm_connection.py
```

6. Set up your social media credentials:
```bash
python setup_credentials.py
```

## Project Structure
```
social_media_manager/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── orchestrator.py
│   │   ├── content_generator.py
│   │   └── platform_manager.py
│   ├── config/          # Configuration files
│   │   ├── __init__.py
│   │   └── llm_config.py
│   └── utils/           # Utility functions
├── tests/              # Test cases
├── examples/           # Example scripts
├── models/            # LLM model directory
├── logs/              # Log files
├── cache/             # Cache directory
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Agent Architecture

### OrchestratorAgent
- Coordinates overall workflow
- Manages task distribution
- Handles error recovery

### ContentGeneratorAgent
- Generates platform-specific content
- Maintains brand voice
- Optimizes for engagement

### PlatformManagerAgent
- Handles platform authentication
- Manages posting schedule
- Tracks post performance

## Local LLM Setup

The system uses a local instance of Mistral-7B served through vLLM for:
- Fast inference with GPU acceleration
- OpenAI-compatible API
- Reduced latency and costs
- Privacy and data control

### LLM Server Configuration
The LLM server runs on `http://localhost:8000` by default and provides:
- `/health` endpoint for status checks
- `/v1/completions` endpoint for text generation
- OpenAI-compatible API interface

Configure the server through environment variables in `.env`:
```
LLM_MODEL=mistral-7b
LLM_API_BASE=http://localhost:8000
LLM_API_TYPE=open_ai
```

## Usage

1. Basic workflow example:
```bash
python examples/basic_workflow.py
```

2. Update social media credentials:
```bash
python setup_credentials.py
```

## Configuration

The system can be configured through environment variables in the `.env` file:

- `LLM_MODEL`: Local LLM model name
- `LLM_API_BASE`: Local LLM API endpoint
- `CACHE_ENABLE`: Enable/disable caching
- Platform-specific API credentials

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Troubleshooting

### LLM Server Issues
1. Check GPU availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Verify server status:
   ```bash
   python test_llm_connection.py
   ```
3. Check logs in `logs/` directory

### Social Media API Issues
1. Verify credentials:
   ```bash
   python setup_credentials.py
   ```
2. Check platform status in logs

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License

## Support
For support:
1. Check the troubleshooting guide
2. Review logs in `logs/` directory
3. Open an issue in the repository

# Deep Research Agent Backend

A FastAPI backend for streaming deep research with multiple AI model support (OpenAI, Anthropic, Kimi K2).

## Features

- 🔄 **Real-time Streaming**: Server-sent events for live research updates
- 🤖 **Multi-Model Support**: OpenAI, Anthropic, and Kimi K2 integration
- 📊 **Performance Metrics**: Model comparison and evaluation
- 🔍 **Deep Research**: Advanced research workflow with multiple stages
- 🚀 **GCP Ready**: Docker configuration for Cloud Run deployment

## Quick Start

### Prerequisites

- Python 3.9+
- pip or uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ShenSeanChen/yt-DeepResearch-Backend.git
cd yt-DeepResearch-Backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### Testing the API

1. Check health status:
```bash
curl http://localhost:8080/health
```

2. Get available models:
```bash
curl http://localhost:8080/models
```

3. Test research endpoint:
```bash
curl -X POST "http://localhost:8080/research/test" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "model": "openai",
    "api_key": "your-api-key-here"
  }'
```

## API Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /models` - Available AI models
- `POST /research/stream` - Stream research process
- `POST /research/test` - Test research parameters

### Analytics Endpoints

- `GET /research/history` - Research history
- `GET /research/comparison` - Model performance comparison
- `DELETE /research/history/{research_id}` - Delete research session

## Streaming Research

The main streaming endpoint accepts:

```json
{
  "query": "Your research question",
  "model": "openai|anthropic|kimi", 
  "api_key": "your-model-api-key"
}
```

Returns Server-Sent Events with real-time updates:

```
data: {"type": "stage_start", "stage": "initialization", "content": "Starting research..."}
data: {"type": "stage_update", "stage": "clarification", "content": "Analyzing scope..."}
data: {"type": "research_complete", "duration": 45.2}
```

## Model Support

### OpenAI
- Model: `gpt-4`
- Requires: `OPENAI_API_KEY` or user-provided key

### Anthropic  
- Model: `claude-3-5-sonnet-20241022`
- Requires: `ANTHROPIC_API_KEY` or user-provided key

### Kimi K2
- Model: `moonshot-v1-128k`
- Requires: `KIMI_API_KEY` or user-provided key

## Docker Deployment

### Local Docker

```bash
docker build -t deep-research-backend .
docker run -p 8080:8080 deep-research-backend
```

### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/deep-research-backend

# Deploy to Cloud Run
gcloud run deploy deep-research-backend \
  --image gcr.io/YOUR_PROJECT/deep-research-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

### Code Structure

```
yt-DeepResearch-Backend/
├── main.py                 # FastAPI application
├── models/                 # Pydantic models
│   └── research_models.py  # API schemas
├── services/               # Business logic
│   ├── deep_research_service.py  # Research workflow
│   └── model_service.py    # Model management
├── utils/                  # Utilities
│   └── metrics.py          # Performance tracking
├── tests/                  # Test suite
└── requirements.txt        # Dependencies
```

## Environment Variables

Optional environment variables for default API keys:

```bash
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key  
KIMI_API_KEY=your-kimi-key
PORT=8080
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

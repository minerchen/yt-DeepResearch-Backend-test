# Copyright 2024
# Directory: yt-DeepResearch-Backend/main.py
"""
FastAPI Backend for Deep Research Agent
Provides streaming API endpoints for deep research with multiple AI model support
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.deep_research_service import DeepResearchService
from services.model_service import ModelService
from models.research_models import (
    ResearchRequest,
    ResearchResponse,
    StreamingEvent,
    ModelComparison,
    ResearchHistory
)
from utils.metrics import MetricsCollector

# Configure logging with Google Standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with comprehensive configuration
app = FastAPI(
    title="Deep Research Agent API",
    description="Streaming deep research API with multiple AI model support (OpenAI, Anthropic, Kimi K2)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
research_service = DeepResearchService()
model_service = ModelService()
metrics_collector = MetricsCollector()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Deep Research Agent API is running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with service status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "research_service": "active",
            "model_service": "active",
            "metrics_collector": "active"
        },
        "supported_models": ["openai", "anthropic", "kimi"]
    }

@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    """
    Stream deep research process with real-time updates
    
    This endpoint provides Server-Sent Events streaming of the research process,
    showing each stage, thinking process, and tool usage in real-time.
    
    Args:
        request: Research request containing query, model, and API key
    
    Returns:
        StreamingResponse: Server-sent events stream
    """
    try:
        # Validate the request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Research query cannot be empty")
        
        if not request.api_key.strip():
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Validate model selection
        available_models = await model_service.get_available_models()
        if request.model not in [model.id for model in available_models["models"]]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model}. Available models: {[m.id for m in available_models['models']]}"
            )
        
        # Create streaming generator
        async def generate_research_stream() -> AsyncGenerator[str, None]:
            research_id = f"research_{int(time.time())}"
            start_time = time.time()
            
            try:
                # Initialize research session
                yield f"data: {json.dumps({'type': 'session_start', 'research_id': research_id, 'timestamp': datetime.utcnow().isoformat(), 'model': request.model, 'query': request.query})}\n\n"
                
                # Stream the research process
                async for event in research_service.stream_research(
                    query=request.query,
                    model=request.model,
                    api_key=request.api_key,
                    research_id=research_id
                ):
                    yield f"data: {json.dumps(event.dict())}\n\n"
                
                # Calculate final metrics
                end_time = time.time()
                duration = end_time - start_time
                
                # Send completion event
                completion_event = {
                    'type': 'research_complete',
                    'research_id': research_id,
                    'duration': duration,
                    'model': request.model,
                    'timestamp': datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(completion_event)}\n\n"
                
                # Store metrics for comparison
                await metrics_collector.store_research_metrics(
                    research_id=research_id,
                    model=request.model,
                    duration=duration,
                    query=request.query
                )
                
            except Exception as e:
                logger.error(f"Error in research stream: {str(e)}")
                error_event = {
                    'type': 'error',
                    'message': str(e),
                    'research_id': research_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
        
        return StreamingResponse(
            generate_research_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting research stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get list of available AI models with their capabilities"""
    return await model_service.get_available_models()

@app.get("/research/history")
async def get_research_history(limit: int = 10):
    """Get research history with performance metrics"""
    return await metrics_collector.get_research_history(limit=limit)

@app.get("/research/comparison")
async def get_model_comparison():
    """Get performance comparison between different models"""
    return await metrics_collector.get_model_comparison()

@app.delete("/research/history/{research_id}")
async def delete_research(research_id: str):
    """Delete a specific research session"""
    success = await metrics_collector.delete_research(research_id)
    if not success:
        raise HTTPException(status_code=404, detail="Research not found")
    return {"message": "Research deleted successfully"}

@app.post("/research/test")
async def test_research_endpoint(request: ResearchRequest):
    """Test endpoint for development and debugging"""
    return {
        "message": "Test endpoint - research parameters received",
        "query": request.query,
        "model": request.model,
        "api_key_length": len(request.api_key) if request.api_key else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    # Disable reload in production for faster startup
    reload = os.getenv("ENVIRONMENT", "production") == "development"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )
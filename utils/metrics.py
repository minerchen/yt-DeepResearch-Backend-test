# Directory: yt-DeepResearch-Backend/utils/metrics.py
"""
Metrics Collection Service - Tracks and compares model performance
Provides evaluation metrics for OpenAI, Anthropic, and Kimi K2 comparison
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

from models.research_models import (
    ModelMetrics, 
    ModelComparison, 
    ResearchHistory
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Service for collecting and analyzing research performance metrics"""
    
    def __init__(self):
        """Initialize metrics collector with in-memory storage"""
        self.research_history: List[Dict] = []
        self.model_metrics: Dict[str, Dict] = {
            "openai": {"requests": [], "total_duration": 0, "success_count": 0},
            "anthropic": {"requests": [], "total_duration": 0, "success_count": 0},
            "kimi": {"requests": [], "total_duration": 0, "success_count": 0}
        }
    
    async def store_research_metrics(
        self,
        research_id: str,
        model: str,
        duration: float,
        query: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Store metrics for a completed research session
        
        Args:
            research_id: Unique research session identifier
            model: AI model used
            duration: Research duration in seconds
            query: Research query
            success: Whether research completed successfully
            error: Error message if applicable
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Store in research history
            research_record = {
                "research_id": research_id,
                "model": model,
                "duration": duration,
                "query": query,
                "success": success,
                "timestamp": timestamp,
                "error": error
            }
            self.research_history.append(research_record)
            
            # Update model-specific metrics
            if model in self.model_metrics:
                self.model_metrics[model]["requests"].append({
                    "duration": duration,
                    "success": success,
                    "timestamp": timestamp
                })
                self.model_metrics[model]["total_duration"] += duration
                if success:
                    self.model_metrics[model]["success_count"] += 1
            
            logger.info(f"Stored metrics for research {research_id} using {model}")
            
        except Exception as e:
            logger.error(f"Error storing research metrics: {str(e)}")
    
    async def get_research_history(self, limit: int = 10) -> Dict:
        """
        Get recent research history
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            Dictionary containing research history
        """
        try:
            # Sort by timestamp (most recent first)
            sorted_history = sorted(
                self.research_history,
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            # Convert to ResearchHistory objects
            history_items = []
            for item in sorted_history[:limit]:
                history_items.append(ResearchHistory(
                    research_id=item["research_id"],
                    query=item["query"],
                    model=item["model"],
                    duration=item["duration"],
                    success=item["success"],
                    timestamp=item["timestamp"],
                    summary=item.get("summary", f"Research using {item['model']}")
                ))
            
            return {
                "history": [item.dict() for item in history_items],
                "total_count": len(self.research_history),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting research history: {str(e)}")
            return {"history": [], "total_count": 0, "error": str(e)}
    
    async def get_model_comparison(self) -> ModelComparison:
        """
        Get performance comparison between different models
        
        Returns:
            ModelComparison with metrics for each model
        """
        try:
            model_metrics_list = []
            total_requests = 0
            
            for model_id, metrics in self.model_metrics.items():
                requests = metrics["requests"]
                request_count = len(requests)
                total_requests += request_count
                
                if request_count > 0:
                    # Calculate average duration
                    avg_duration = metrics["total_duration"] / request_count
                    
                    # Calculate success rate
                    success_rate = (metrics["success_count"] / request_count) * 100
                    
                    # Get last used timestamp
                    last_used = max(req["timestamp"] for req in requests) if requests else None
                else:
                    avg_duration = 0.0
                    success_rate = 0.0
                    last_used = None
                
                model_metrics_list.append(ModelMetrics(
                    model=model_id,
                    total_requests=request_count,
                    average_duration=round(avg_duration, 2),
                    success_rate=round(success_rate, 2),
                    last_used=last_used
                ))
            
            return ModelComparison(
                models=model_metrics_list,
                total_requests=total_requests,
                generated_at=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating model comparison: {str(e)}")
            return ModelComparison(
                models=[],
                total_requests=0,
                generated_at=datetime.utcnow().isoformat()
            )
    
    async def delete_research(self, research_id: str) -> bool:
        """
        Delete a specific research session
        
        Args:
            research_id: Research session to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            # Find and remove from history
            original_count = len(self.research_history)
            self.research_history = [
                item for item in self.research_history 
                if item["research_id"] != research_id
            ]
            
            deleted = len(self.research_history) < original_count
            
            if deleted:
                logger.info(f"Deleted research session {research_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting research {research_id}: {str(e)}")
            return False
    
    async def get_detailed_metrics(self, model: str) -> Dict:
        """
        Get detailed metrics for a specific model
        
        Args:
            model: Model identifier
            
        Returns:
            Detailed metrics dictionary
        """
        try:
            if model not in self.model_metrics:
                return {"error": f"Model {model} not found"}
            
            metrics = self.model_metrics[model]
            requests = metrics["requests"]
            
            if not requests:
                return {
                    "model": model,
                    "total_requests": 0,
                    "metrics": {}
                }
            
            # Calculate detailed statistics
            durations = [req["duration"] for req in requests]
            successful_requests = [req for req in requests if req["success"]]
            
            return {
                "model": model,
                "total_requests": len(requests),
                "successful_requests": len(successful_requests),
                "success_rate": (len(successful_requests) / len(requests)) * 100,
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations),
                "recent_requests": requests[-5:] if len(requests) >= 5 else requests
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed metrics for {model}: {str(e)}")
            return {"error": str(e)}

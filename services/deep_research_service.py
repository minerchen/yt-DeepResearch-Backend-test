# Directory: yt-DeepResearch-Backend/services/deep_research_service.py
"""
Deep Research Service - Integrates the original deep_researcher.py with streaming capabilities
Provides real-time streaming of research workflow stages and thinking processes
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional

# Add the open_deep_research to Python path
sys.path.append('/Users/shenseanchen/Desktop/Dev/yt-DeepResearchAgent/open_deep_research/src')

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration
from open_deep_research.state import AgentInputState
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from models.research_models import StreamingEvent, ResearchStage
from services.model_service import ModelService

logger = logging.getLogger(__name__)


class DeepResearchService:
    """Service for handling deep research operations with streaming support"""
    
    def __init__(self):
        """Initialize the deep research service"""
        self.model_service = ModelService()
    
    async def stream_research(
        self,
        query: str,
        model: str,
        api_key: str,
        research_id: str
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Stream the deep research process with real-time updates
        
        Args:
            query: Research question/topic
            model: AI model to use (openai, anthropic, kimi)
            api_key: User's API key for the selected model
            research_id: Unique identifier for this research session
        
        Yields:
            StreamingEvent: Real-time updates about the research progress
        """
        current_stage = ResearchStage.INITIALIZATION
        
        try:
            # Configure the research workflow
            config = await self._create_research_config(model, api_key)
            
            # Create initial state
            initial_state = AgentInputState(
                messages=[HumanMessage(content=query)]
            )
            
            # Yield initial event
            yield StreamingEvent(
                type="stage_start",
                stage=ResearchStage.INITIALIZATION,
                content=f"🚀 Starting deep research for: {query}",
                timestamp=datetime.utcnow().isoformat(),
                research_id=research_id,
                model=model,
                metadata={"query": query, "model_config": model}
            )
            
            # Stream the research workflow
            node_count = 0
            
            async for chunk in deep_researcher.astream(
                initial_state,
                config=config,
                stream_mode="updates"
            ):
                node_count += 1
                
                # Process each chunk and convert to streaming event
                for node_name, node_data in chunk.items():
                    event = await self._process_workflow_node(
                        node_name, node_data, research_id, model, node_count
                    )
                    
                    if event:
                        current_stage = event.stage or current_stage
                        yield event
            
            # Final completion event
            yield StreamingEvent(
                type="stage_complete",
                stage=ResearchStage.COMPLETED,
                content="✅ Deep research completed successfully!",
                timestamp=datetime.utcnow().isoformat(),
                research_id=research_id,
                model=model,
                metadata={"total_nodes": node_count}
            )
                    
        except Exception as e:
            logger.error(f"Error in stream_research: {str(e)}")
            yield StreamingEvent(
                type="error",
                stage=current_stage,
                content=f"❌ Error occurred: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
                research_id=research_id,
                model=model,
                error=str(e)
            )
    
    async def _create_research_config(self, model: str, api_key: str) -> RunnableConfig:
        """
        Create LangChain configuration for the research workflow
        
        Args:
            model: Model identifier
            api_key: User's API key
            
        Returns:
            RunnableConfig for the research workflow
        """
        try:
            # Get model mapping
            model_mapping = self.model_service.get_model_provider_mapping()
            langchain_model = model_mapping.get(model, "gpt-4")
            
            # Create configuration
            config = RunnableConfig(
                configurable={
                    "research_model": langchain_model,
                    "research_model_max_tokens": 4000,
                    "allow_clarification": True,
                    "max_structured_output_retries": 3,
                    "search_api": "tavily",  # Default search API
                    "max_research_iterations": 5,
                    "max_researchers": 3
                },
                metadata={
                    "user_api_key": api_key,
                    "model_type": model
                }
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating research config: {str(e)}")
            raise
    
    async def _process_workflow_node(
        self,
        node_name: str,
        node_data: Any,
        research_id: str,
        model: str,
        node_count: int
    ) -> Optional[StreamingEvent]:
        """
        Process a workflow node and convert to streaming event
        
        Args:
            node_name: Name of the workflow node
            node_data: Data from the workflow node
            research_id: Research session ID
            model: Model being used
            node_count: Current node number
            
        Returns:
            StreamingEvent or None
        """
        try:
            # Map node names to research stages
            stage_mapping = {
                "clarify_with_user": ResearchStage.CLARIFICATION,
                "write_research_brief": ResearchStage.RESEARCH_BRIEF,
                "research_supervisor": ResearchStage.RESEARCH_EXECUTION,
                "final_report_generation": ResearchStage.FINAL_REPORT
            }
            
            stage = stage_mapping.get(node_name, ResearchStage.RESEARCH_EXECUTION)
            
            # Create content based on node type
            content = await self._generate_node_content(node_name, node_data, node_count)
            
            return StreamingEvent(
                type="stage_update",
                stage=stage,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
                research_id=research_id,
                model=model,
                metadata={
                    "node_name": node_name,
                    "node_count": node_count,
                    "has_messages": hasattr(node_data, 'messages') if hasattr(node_data, '__dict__') else False
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing workflow node {node_name}: {str(e)}")
            return StreamingEvent(
                type="error",
                stage=ResearchStage.ERROR,
                content=f"Error processing {node_name}: {str(e)}",
                timestamp=datetime.utcnow().isoformat(),
                research_id=research_id,
                model=model,
                error=str(e)
            )
    
    async def _generate_node_content(self, node_name: str, node_data: Any, node_count: int) -> str:
        """
        Generate human-readable content for a workflow node
        
        Args:
            node_name: Name of the workflow node
            node_data: Data from the workflow node
            node_count: Current node number
            
        Returns:
            Human-readable content string
        """
        try:
            content_templates = {
                "clarify_with_user": f"🔍 Step {node_count}: Analyzing research scope and clarifying requirements",
                "write_research_brief": f"📝 Step {node_count}: Creating comprehensive research brief and strategy",
                "research_supervisor": f"🔬 Step {node_count}: Conducting deep research using multiple tools and sources",
                "final_report_generation": f"📊 Step {node_count}: Generating final research report with findings and analysis"
            }
            
            base_content = content_templates.get(
                node_name, 
                f"⚙️ Step {node_count}: Processing {node_name.replace('_', ' ').title()}"
            )
            
            # Add additional context if available
            if hasattr(node_data, 'messages') and node_data.messages:
                last_message = node_data.messages[-1]
                if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                    if len(last_message.content) > 100:
                        preview = last_message.content[:100] + "..."
                        base_content += f"\n💭 Thinking: {preview}"
            
            return base_content
            
        except Exception as e:
            logger.error(f"Error generating node content: {str(e)}")
            return f"⚙️ Step {node_count}: Processing {node_name}"

# Directory: yt-DeepResearch-Backend/services/deep_research_service.py
"""
Deep Research Service - Integrates the original deep_researcher.py with streaming capabilities
Provides real-time streaming of research workflow stages and thinking processes
"""

import asyncio
import json
import logging
import re
import sys
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, List

# Set environment variable to get API keys from config
os.environ["GET_API_KEYS_FROM_CONFIG"] = "true"

# Add the open_deep_research to Python path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
                logger.info(f"Processing chunk {node_count}: {list(chunk.keys())}")
                
                # Process each chunk and convert to streaming event
                for node_name, node_data in chunk.items():
                    event = await self._process_workflow_node(
                        node_name, node_data, research_id, model, node_count
                    )
                    
                    if event:
                        current_stage = event.stage or current_stage
                        yield event
                        
                    # Special handling for research_supervisor chunk to show research progress
                    if node_name == "research_supervisor" and node_data:
                        async for research_event in self._process_research_supervisor_data(
                            node_data, research_id, model, node_count
                        ):
                            yield research_event
            
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
            
            # Create API keys configuration
            api_keys = {}
            if model == "openai":
                api_keys["OPENAI_API_KEY"] = api_key
            elif model == "anthropic":
                api_keys["ANTHROPIC_API_KEY"] = api_key
            elif model == "kimi":
                api_keys["KIMI_API_KEY"] = api_key
            
            # Create configuration - use user's chosen model for ALL model operations
            config = RunnableConfig(
                configurable={
                    "research_model": langchain_model,
                    "research_model_max_tokens": 4000,
                    "final_report_model": langchain_model,  # Use same model for final report
                    "final_report_model_max_tokens": 8000,
                    "compression_model": langchain_model,  # Use same model for compression
                    "compression_model_max_tokens": 4000,
                    "summarization_model": langchain_model,  # Use same model for summarization
                    "summarization_model_max_tokens": 4000,
                    "allow_clarification": False,
                    "max_structured_output_retries": 3,
                    "search_api": "anthropic",  # Use Anthropic search API
                    "max_research_iterations": 5,
                    "max_researchers": 3,
                    "apiKeys": api_keys
                },
                metadata={
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
        Generate human-readable content for a workflow node with comprehensive data extraction
        
        Args:
            node_name: Name of the workflow node
            node_data: Data from the workflow node
            node_count: Current node number
            
        Returns:
            Human-readable content string with actual AI messages and content
        """
        try:
            # Log the structure for debugging
            logger.info(f"Processing node {node_name} with data type: {type(node_data)}")
            if hasattr(node_data, '__dict__'):
                logger.info(f"Node data attributes: {list(node_data.__dict__.keys())}")
            
            # Extract actual content based on node type
            extracted_content = ""
            
            if node_name == "clarify_with_user":
                extracted_content = f"🔍 Step {node_count}: Analyzing research scope and clarifying requirements"
                
                # Extract all messages from clarification
                ai_messages = self._extract_ai_messages(node_data)
                if ai_messages:
                    extracted_content += f"\n\n💭 AI Clarification Process:"
                    for i, msg in enumerate(ai_messages[:3]):  # Show up to 3 messages
                        extracted_content += f"\n🤖 Message {i+1}: {msg}"
                else:
                    # Fallback: try to extract any text content
                    fallback_content = self._extract_text_content(node_data)
                    if fallback_content:
                        extracted_content += f"\n💭 AI Decision: {fallback_content}"
                
            elif node_name == "write_research_brief":
                extracted_content = f"📝 Step {node_count}: Creating comprehensive research brief and strategy"
                
                # Extract AI messages from research brief creation
                ai_messages = self._extract_ai_messages(node_data)
                if ai_messages:
                    extracted_content += f"\n\n📋 AI Research Brief Generation:"
                    for i, msg in enumerate(ai_messages[:2]):  # Show up to 2 messages
                        extracted_content += f"\n🤖 Brief {i+1}: {msg}"
                
                # Also look for specific research brief attributes
                if hasattr(node_data, 'research_brief') and node_data.research_brief:
                    brief_content = str(node_data.research_brief)[:300] + "..." if len(str(node_data.research_brief)) > 300 else str(node_data.research_brief)
                    extracted_content += f"\n📄 Final Brief: {brief_content}"
                
                # Fallback content extraction
                if not ai_messages:
                    fallback_content = self._extract_text_content(node_data)
                    if fallback_content:
                        extracted_content += f"\n📋 Research Strategy: {fallback_content}"
                
            elif node_name == "research_supervisor":
                extracted_content = f"🔬 Step {node_count}: Conducting deep research using multiple tools and sources"
                
                # Extract research findings
                if hasattr(node_data, 'notes') and node_data.notes:
                    findings_count = len(node_data.notes)
                    extracted_content += f"\n📊 Found {findings_count} research findings"
                    # Show first few findings with source extraction
                    for i, note in enumerate(node_data.notes[:3]):
                        if note and len(str(note)) > 20:
                            note_content = str(note)
                            # Extract sources from the note
                            sources = self._extract_sources_from_text(note_content)
                            note_preview = note_content[:150] + "..." if len(note_content) > 150 else note_content
                            extracted_content += f"\n🔍 Finding {i+1}: {note_preview}"
                            if sources:
                                extracted_content += f"\n📎 Sources: {', '.join(sources[:2])}"
                
                # Extract compressed research
                if hasattr(node_data, 'compressed_research') and node_data.compressed_research:
                    research_summary = str(node_data.compressed_research)[:200] + "..." if len(str(node_data.compressed_research)) > 200 else str(node_data.compressed_research)
                    extracted_content += f"\n📝 Research Summary: {research_summary}"
                
            elif node_name == "final_report_generation":
                extracted_content = f"📊 Step {node_count}: Generating final research report with findings and analysis"
                
                # Extract AI messages from final report generation
                ai_messages = self._extract_ai_messages(node_data)
                if ai_messages:
                    extracted_content += f"\n\n📄 AI Report Generation:"
                    for i, msg in enumerate(ai_messages[:2]):  # Show up to 2 messages
                        extracted_content += f"\n🤖 Report {i+1}: {msg}"
                
                # Extract final report content
                if hasattr(node_data, 'final_report') and node_data.final_report:
                    report_preview = str(node_data.final_report)[:400] + "..." if len(str(node_data.final_report)) > 400 else str(node_data.final_report)
                    extracted_content += f"\n📋 Generated Report: {report_preview}"
                
                # Fallback content extraction
                if not ai_messages:
                    fallback_content = self._extract_text_content(node_data)
                    if fallback_content:
                        extracted_content += f"\n📄 Final Report: {fallback_content}"
            
            else:
                extracted_content = f"⚙️ Step {node_count}: Processing {node_name.replace('_', ' ').title()}"
            
            # Always try to extract general message content if we haven't found specific content
            if "\n" not in extracted_content and hasattr(node_data, 'messages') and node_data.messages:
                for msg in node_data.messages:
                    if hasattr(msg, 'content') and isinstance(msg.content, str) and len(msg.content) > 50:
                        preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                        extracted_content += f"\n💭 Content: {preview}"
                        break
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error generating node content for {node_name}: {str(e)}")
            return f"⚙️ Step {node_count}: Processing {node_name}"
    
    def _extract_ai_messages(self, node_data: Any) -> List[str]:
        """
        Extract AI messages from node data
        
        Args:
            node_data: Data from the workflow node
            
        Returns:
            List of AI message contents
        """
        messages = []
        
        try:
            # Check for messages attribute
            if hasattr(node_data, 'messages') and node_data.messages:
                for msg in node_data.messages:
                    if hasattr(msg, 'content'):
                        content = str(msg.content)
                        # Filter for substantial AI responses (not just system messages)
                        if len(content) > 20 and not content.startswith('Human:'):
                            # Truncate very long messages
                            if len(content) > 500:
                                content = content[:500] + "..."
                            messages.append(content)
            
            # Also check for direct content in various possible attributes
            content_attributes = ['content', 'response', 'output', 'result', 'text']
            for attr in content_attributes:
                if hasattr(node_data, attr):
                    attr_value = getattr(node_data, attr)
                    if attr_value and isinstance(attr_value, str) and len(attr_value) > 20:
                        if len(attr_value) > 500:
                            attr_value = attr_value[:500] + "..."
                        messages.append(attr_value)
                        break
            
            return messages[:5]  # Limit to 5 messages max
            
        except Exception as e:
            logger.error(f"Error extracting AI messages: {str(e)}")
            return []
    
    def _extract_text_content(self, node_data: Any) -> str:
        """
        Extract any meaningful text content from node data as fallback
        
        Args:
            node_data: Data from the workflow node
            
        Returns:
            Extracted text content or empty string
        """
        try:
            # Try various ways to extract text content
            if hasattr(node_data, 'messages') and node_data.messages:
                # Get the last substantial message
                for msg in reversed(node_data.messages):
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        if len(msg.content) > 50:
                            return msg.content[:300] + ("..." if len(msg.content) > 300 else "")
            
            # Try converting the whole object to string as last resort
            if hasattr(node_data, '__dict__'):
                data_str = str(node_data.__dict__)
                if len(data_str) > 100:
                    return data_str[:200] + "..."
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return ""
    
    def _extract_sources_from_text(self, text: str) -> List[str]:
        """
        Extract sources and URLs from research text
        
        Args:
            text: Text content to extract sources from
            
        Returns:
            List of extracted sources
        """
        sources = []
        
        try:
            # Extract URLs
            url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
            urls = re.findall(url_pattern, text)
            sources.extend(urls)
            
            # Extract source patterns
            source_patterns = [
                r'SOURCE:\s*([^\n]+)',
                r'Source:\s*([^\n]+)', 
                r'from\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'according to\s+([^\n,.]+)',
                r'cited from\s+([^\n,.]+)',
                r'reference:\s*([^\n]+)',
                r'via\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ]
            
            for pattern in source_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                sources.extend([match.strip() for match in matches if len(match.strip()) > 3])
            
            # Remove duplicates and filter
            unique_sources = list(set(sources))
            return [source for source in unique_sources if len(source) > 3][:5]  # Limit to 5 sources
            
        except Exception as e:
            logger.error(f"Error extracting sources: {str(e)}")
            return []
    
    async def _process_research_supervisor_data(
        self,
        node_data: Any,
        research_id: str,
        model: str,
        node_count: int
    ) -> None:
        """
        Process research supervisor data to extract and stream research findings
        
        Args:
            node_data: Data from the research supervisor node
            research_id: Research session ID
            model: Model being used
            node_count: Current node number
        """
        try:
            # Check if we have research findings
            if hasattr(node_data, 'notes') and node_data.notes:
                for i, note in enumerate(node_data.notes):
                    if note and len(str(note)) > 50:  # Only show substantial content
                        yield StreamingEvent(
                            type="research_finding",
                            stage=ResearchStage.RESEARCH_EXECUTION,
                            content=f"🔍 Research Finding {i+1}: {str(note)[:200]}..." if len(str(note)) > 200 else f"🔍 Research Finding {i+1}: {str(note)}",
                            timestamp=datetime.utcnow().isoformat(),
                            research_id=research_id,
                            model=model,
                            metadata={
                                "finding_index": i+1,
                                "finding_length": len(str(note)),
                                "node_count": node_count
                            }
                        )
            
            # Check for compressed research
            if hasattr(node_data, 'compressed_research') and node_data.compressed_research:
                yield StreamingEvent(
                    type="research_summary",
                    stage=ResearchStage.RESEARCH_EXECUTION,
                    content=f"📊 Research Summary: {str(node_data.compressed_research)[:300]}..." if len(str(node_data.compressed_research)) > 300 else f"📊 Research Summary: {str(node_data.compressed_research)}",
                    timestamp=datetime.utcnow().isoformat(),
                    research_id=research_id,
                    model=model,
                    metadata={
                        "summary_length": len(str(node_data.compressed_research)),
                        "node_count": node_count
                    }
                )
                
        except Exception as e:
            logger.error(f"Error processing research supervisor data: {str(e)}")
            # Don't yield error events for this as it's supplementary

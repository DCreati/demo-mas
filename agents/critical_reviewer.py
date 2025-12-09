from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState
from utils.logger import logger, format_agent_message
from utils.prompts import build_critical_prompt
from utils.model_factory import create_llm


class CriticalReviewerAgent:
    def __init__(self, model_name: str = "llama3.1:8b", local: int = 1):
        self.name = "Critical Reviewer"
        self.model_name = model_name
        
        self.llm = create_llm(
            model_name=model_name,
            local=local,
            temperature=0.6,  # Higher creativity for identifying issues
            num_predict=800
        )
        
        logger.info(f"{self.name} agent ready")
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        logger.agent_start(self.name, "Performing Critical Analysis")
        
        try:
            paper_abstract = state.get("paper_abstract", "")
            lit_findings = state.get("literature_findings", "")
            tech_analysis = state.get("technical_analysis", "")
            
            if not paper_abstract:
                logger.warning("No paper abstract available")
                return state
            
            context_sources = []
            if lit_findings:
                context_sources.append("literature review")
            if tech_analysis:
                context_sources.append("technical analysis")
            
            if context_sources:
                logger.info(f"Integrating insights from: {', '.join(context_sources)}")
            else:
                logger.warning("Limited context available")
            
            review = self._perform_critical_review(state)
            
            logger.reasoning(f"Performed critical evaluation by synthesizing {len(context_sources)} "
                           f"previous analyses. Identified limitations, questioned assumptions, "
                           f"and suggested concrete improvements.")
            
            updated_state = state.copy()
            updated_state["critical_review"] = review
            
            message = format_agent_message(
                agent_name=self.name,
                content=f"Completed critical review. Identified limitations and suggested improvements based on holistic analysis.",
                action="critical_review"
            )
            current_messages = state.get("messages", [])
            updated_state["messages"] = current_messages + [message]
            
            logger.state_update("critical_review", review[:150])
            logger.success("Critical review complete")
            
            updated_state["next_agent"] = "supervisor"
                    
            return updated_state
            
        except Exception as e:
            logger.error(f"Critical review failed: {str(e)}")
            return state
    
    def _perform_critical_review(self, state: AgentState) -> str:
        system_prompt = build_critical_prompt(state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Provide your critical review following the specified format. "
                                "Consider both literature and technical perspectives.")
        ]
        
        logger.info("Running the critical review")
        response = self.llm.invoke(messages)
        # Log the LLM reasoning (first 500 chars) for traceability
        try:
            logger.reasoning(response.content[:500])
        except Exception:
            pass

        return response.content
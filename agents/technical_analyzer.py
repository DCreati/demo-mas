from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState
from utils.logger import logger, format_agent_message
from utils.prompts import build_technical_prompt
from utils.model_factory import create_llm


class TechnicalAnalyzerAgent:
    
    def __init__(self, model_name: str = "llama3.1:8b", local: int = 1):
        self.name = "Technical Analyzer"
        self.model_name = model_name
        
        self.llm = create_llm(
            model_name=model_name,
            local=local,
            temperature=0.4,  # Balanced for technical precision
            num_predict=800
        )
        
        logger.info(f"{self.name} agent ready")
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        logger.agent_start(self.name, "Evaluating Technical Methodology")
        
        try:
            paper_abstract = state.get("paper_abstract", "")
            literature_context = state.get("literature_findings", "")
            
            if not paper_abstract:
                logger.warning("No paper abstract available")
                return state
            
            if literature_context:
                logger.info("Building on literature review context")
            else:
                logger.warning("No literature context available - proceeding anyway")
            
            analysis = self._analyze_technical_approach(state)
            
            logger.reasoning(f"Evaluated technical methodology, assessed soundness, "
                           f"identified strengths and potential concerns based on "
                           f"both paper content and literature context.")
            
            updated_state = state.copy()
            updated_state["technical_analysis"] = analysis
            
            message = format_agent_message(
                agent_name=self.name,
                content=f"Completed technical analysis. Assessed methodology soundness and identified key technical aspects.",
                action="technical_analysis"
            )
            current_messages = state.get("messages", [])
            updated_state["messages"] = current_messages + [message]
            
            logger.state_update("technical_analysis", analysis[:150])
            logger.success("Technical analysis complete")
            
            updated_state["next_agent"] = "supervisor"
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return state
    
    def _analyze_technical_approach(self, state: AgentState) -> str:
        system_prompt = build_technical_prompt(state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Provide your technical analysis following the specified format, building on the literature context.")
        ]
        
        logger.info("Running the technical analysis")
        response = self.llm.invoke(messages)
        try:
            logger.reasoning(response.content[:500])
        except Exception:
            pass

        return response.content
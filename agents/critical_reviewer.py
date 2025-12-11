import json
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
            temperature=0.4,  # Balanced temperature for fair quality assessment
            num_predict=600
        )
        
        logger.info(f"{self.name} agent ready - Quality Assessor role")
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        logger.agent_start(self.name, "Quality Assessment and Rerun Recommendation")
        
        try:
            paper_abstract = state.get("paper_abstract", "")
            lit_findings = state.get("literature_findings", "")
            tech_analysis = state.get("technical_analysis", "")
            
            if not paper_abstract:
                logger.warning("No paper abstract available")
                return state
            
            if not lit_findings or not tech_analysis:
                logger.warning("Cannot perform quality assessment: literature or technical analysis missing")
                return state
            
            logger.info("Assessing quality of literature review and technical analysis")
            logger.info("Will determine if reruns are needed or if workflow should proceed to synthesis")
            
            evaluation_json = self._evaluate_quality_and_reruns(state)
            
            updated_state = state.copy()
            updated_state["critical_evaluation"] = evaluation_json
            
            # Parse the JSON to extract rerun recommendations
            try:
                parsed_eval = json.loads(evaluation_json)
                needs_rerun = parsed_eval.get("needs_rerun", [])
                
                if needs_rerun:
                    logger.warning(f"Critical Reviewer recommends reruns: {needs_rerun}")
                    updated_state["needs_rerun"] = needs_rerun
                    # Track rerun counts
                    for agent in needs_rerun:
                        if agent == "literature_reviewer":
                            updated_state["literature_rerun_count"] = state.get("literature_rerun_count", 0) + 1
                        elif agent == "technical_analyzer":
                            updated_state["technical_rerun_count"] = state.get("technical_rerun_count", 0) + 1
                else:
                    logger.success("Critical Reviewer assessment: Quality is acceptable, proceeding to synthesis")
                    updated_state["needs_rerun"] = []
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse critical evaluation JSON: {str(e)}")
                logger.warning("Treating evaluation as acceptable and proceeding to synthesis")
                updated_state["needs_rerun"] = []
            
            message = format_agent_message(
                agent_name=self.name,
                content=f"Quality assessment complete. Rerun recommendations: {updated_state.get('needs_rerun', [])}",
                action="quality_assessment"
            )
            current_messages = state.get("messages", [])
            updated_state["messages"] = current_messages + [message]
            
            logger.state_update("critical_evaluation", "Quality assessment stored")
            logger.success("Critical assessment complete")
            
            # Always route back to supervisor for next decision
            updated_state["next_agent"] = "supervisor"
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Critical assessment failed: {str(e)}")
            return state
    
    def _evaluate_quality_and_reruns(self, state: AgentState) -> str:
        """
        Evaluates the quality of literature and technical analyses.
        Returns JSON with quality assessment and rerun recommendations.
        """
        system_prompt = build_critical_prompt(state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content="Evaluate the quality of the literature review and technical analysis. "
                        "Determine if either agent should be re-run to improve coverage or accuracy. "
                        "Return ONLY valid JSON as specified in the prompt."
            )
        ]
        
        logger.info("Running quality assessment via LLM")
        response = self.llm.invoke(messages)
        
        evaluation_text = response.content
        
        try:
            logger.reasoning(evaluation_text[:400])
        except Exception:
            pass
        
        return evaluation_text
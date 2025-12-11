from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState
from utils.logger import logger, format_agent_message
from utils.prompts import build_synthesis_prompt
from utils.model_factory import create_llm


class SynthesisAgent:
    
    def __init__(self, model_name: str = "llama3.1:8b", local: int = 1):
        self.name = "Synthesis Agent"
        self.model_name = model_name
        
        self.llm = create_llm(
            model_name=model_name,
            local=local,
            temperature=0.5,  # Balanced for coherent synthesis
            num_predict=1200  # Longer for comprehensive report
        )
        
        logger.info(f"{self.name} is ready")
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        logger.agent_start(self.name, "Synthesizing Final Report")
        
        try:
            paper_abstract = state.get("paper_abstract", "")
            lit_findings = state.get("literature_findings", "")
            tech_analysis = state.get("technical_analysis", "")
            critical_review = state.get("critical_review", "")
            
            if not all([paper_abstract, lit_findings, tech_analysis, critical_review]):
                missing = []
                if not lit_findings:
                    missing.append("literature review")
                if not tech_analysis:
                    missing.append("technical analysis")
                if not critical_review:
                    missing.append("critical review")
                
                # logger.warning(f"Incomplete analyses. Missing: {', '.join(missing)}")
                logger.info("Proceeding with the available information")
            
            total_content = sum([
                len(lit_findings),
                len(tech_analysis),
                len(critical_review)
            ])
            logger.info(f"Integrating {total_content} characters from three agents")
            
            final_report = self._synthesize_report(state)
            
            logger.reasoning(
                f"Synthesized insights from {3} specialized agents into a unified review. "
                f"This demonstrates EMERGENT BEHAVIOR - the final report's quality and "
                f"structure emerges from simple agent interactions, not explicit programming. "
                f"Each agent contributed specialized analysis; synthesis creates holistic value."
            )
            
            updated_state = state.copy()
            updated_state["final_report"] = final_report
            updated_state["analysis_complete"] = True
            
            message = format_agent_message(
                agent_name=self.name,
                content=f"Completed final synthesis. Generated comprehensive review report integrating all agent findings.",
                action="synthesis_complete"
            )
            current_messages = state.get("messages", [])
            updated_state["messages"] = current_messages + [message]
            
            logger.state_update("final_report", "Complete synthesis generated")
            logger.success("Synthesis complete - final report generated")
            
            updated_state["next_agent"] = "FINISH"
            
            logger.final_output(final_report)
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return state
    
    def _synthesize_report(self, state: AgentState) -> str:
        system_prompt = build_synthesis_prompt(state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Create the final comprehensive review report by synthesizing "
                                "all the analyses. Follow the specified format and provide a "
                                "balanced, professional assessment.")
        ]
        
        logger.info("Running the final synthesis")
        logger.info("Pulling together the remaining context for the report")
        
        response = self.llm.invoke(messages)
        try:
            logger.reasoning(response.content[:500])
        except Exception:
            pass

        return response.content
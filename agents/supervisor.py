import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import AgentState
from utils.logger import logger, format_agent_message
from utils.prompts import build_supervisor_prompt
from utils.model_factory import create_llm


class SupervisorAgent:
    
    def __init__(self, model_name: str = "llama3.1:8b", local: int = 1):
        self.name = "Supervisor"
        self.model_name = model_name
        
        self.llm = create_llm(
            model_name=model_name,
            local=local,
            temperature=0.3,  # Lower temperature for consistent decisions
            num_predict=500   # Limit output length
        )
        
        logger.info(f"{self.name} agent ready")
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        logger.agent_start(self.name, "Hierarchical Coordinator")
        
        try:
            iteration = state.get("iteration_count", 0) + 1
            logger.info(f"Reviewing workflow state (iteration {iteration})")
            
            if iteration > 10:
                logger.warning("Maximum iterations reached. Forcing completion.")
                return self._force_completion(state)
            
            reasoning_output = self._make_routing_decision(state)
            
            decision = self._parse_decision(reasoning_output)
            
            logger.reasoning(decision["reasoning"])
            logger.decision(
                f"Route to {decision['next_agent']}", 
                f"Priority: {decision.get('priority', 'medium')}"
            )
            
            updated_state = state.copy()
            updated_state["next_agent"] = decision["next_agent"]
            updated_state["iteration_count"] = iteration
            
            supervisor_message = format_agent_message(
                agent_name=self.name,
                content=f"Routing decision: {decision['next_agent']}. {decision['reasoning'][:100]}",
                action="route"
            )
            current_messages = state.get("messages", [])
            updated_state["messages"] = current_messages + [supervisor_message]
            
            if decision["next_agent"] == "FINISH":
                updated_state["analysis_complete"] = True
                logger.success("Workflow marked as complete by Supervisor")
            
            logger.state_update("next_agent", decision["next_agent"])
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Supervisor execution failed: {str(e)}")
            return self._fallback_routing(state)
    
    def _make_routing_decision(self, state: AgentState) -> str:
        system_prompt = build_supervisor_prompt(state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze the current state and decide the next agent to execute. Provide your response in the JSON format specified.")
        ]
        
        logger.info("Consulting the LLM for a routing decision")
        response = self.llm.invoke(messages)
        
        return response.content
    
    def _parse_decision(self, llm_output: str) -> Dict[str, str]:
        try:
            if "{" in llm_output and "}" in llm_output:
                json_start = llm_output.find("{")
                json_end = llm_output.rfind("}") + 1
                json_str = llm_output[json_start:json_end]
                decision = json.loads(json_str)
                
                if "next_agent" in decision:
                    return decision
        except:
            pass
        
        logger.warning("Could not parse JSON, using fallback extraction")
        decision = self._extract_decision_from_text(llm_output)
        return decision
    
    def _extract_decision_from_text(self, text: str) -> Dict[str, str]:
        text_lower = text.lower()
        
        if "literature" in text_lower and "review" in text_lower:
            next_agent = "literature_reviewer"
        elif "technical" in text_lower and "analy" in text_lower:
            next_agent = "technical_analyzer"
        elif "critical" in text_lower or "review" in text_lower:
            next_agent = "critical_reviewer"
        elif "synthesis" in text_lower or "final" in text_lower:
            next_agent = "synthesis"
        elif "finish" in text_lower or "complete" in text_lower:
            next_agent = "FINISH"
        else:
            next_agent = self._get_default_next_agent(text)
        
        return {
            "reasoning": text[:200],
            "next_agent": next_agent,
            "priority": "medium"
        }
    
    def _get_default_next_agent(self, text: str) -> str:
        return "literature_reviewer"
    
    def _fallback_routing(self, state: AgentState) -> Dict[str, Any]:
        logger.warning("Using fallback routing logic")
        
        updated_state = state.copy()
        
        if not state.get("literature_findings"):
            updated_state["next_agent"] = "literature_reviewer"
        elif not state.get("technical_analysis"):
            updated_state["next_agent"] = "technical_analyzer"
        elif not state.get("critical_review"):
            updated_state["next_agent"] = "critical_reviewer"
        elif not state.get("final_report"):
            updated_state["next_agent"] = "synthesis"
        else:
            updated_state["next_agent"] = "FINISH"
            updated_state["analysis_complete"] = True
        
        return updated_state
    
    def _force_completion(self, state: AgentState) -> Dict[str, Any]:
        updated_state = state.copy()
        updated_state["next_agent"] = "FINISH"
        updated_state["analysis_complete"] = True
        
        message = format_agent_message(
            agent_name=self.name,
            content="Maximum iterations reached. Forcing workflow completion.",
            action="force_complete"
        )
        updated_state["messages"] = state.get("messages", []) + [message]
        
        return updated_state


def route_to_next_agent(state: AgentState) -> str:
    next_agent = state.get("next_agent", "FINISH")
    
    agent_map = {
        "literature_reviewer": "literature_reviewer",
        "technical_analyzer": "technical_analyzer",
        "critical_reviewer": "critical_reviewer",
        "synthesis": "synthesis",
        "FINISH": "END"
    }
    
    return agent_map.get(next_agent, "END")
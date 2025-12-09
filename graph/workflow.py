from typing import Literal
from langgraph.graph import StateGraph, END
from graph.state import AgentState

from agents.supervisor import SupervisorAgent, route_to_next_agent
from agents.literature_reviewer import LiteratureReviewerAgent
from agents.technical_analyzer import TechnicalAnalyzerAgent
from agents.critical_reviewer import CriticalReviewerAgent
from agents.synthesis_agent import SynthesisAgent

from utils.logger import logger


def create_research_workflow(model_name: str = "llama3.1:8b", local: int = 1) -> StateGraph:
    logger.info("Building the multi-agent workflow graph")
    
    supervisor = SupervisorAgent(model_name, local)
    literature_reviewer = LiteratureReviewerAgent(model_name, local)
    technical_analyzer = TechnicalAnalyzerAgent(model_name, local)
    critical_reviewer = CriticalReviewerAgent(model_name, local)
    synthesis_agent = SynthesisAgent(model_name, local)
    
    logger.info("All agents initialized")
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor.execute)
    workflow.add_node("literature_reviewer", literature_reviewer.execute)
    workflow.add_node("technical_analyzer", technical_analyzer.execute)
    workflow.add_node("critical_reviewer", critical_reviewer.execute)
    workflow.add_node("synthesis", synthesis_agent.execute)
    
    logger.info("Graph nodes (agents) added")
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_to_next_agent,
        {
            "literature_reviewer": "literature_reviewer",
            "technical_analyzer": "technical_analyzer",
            "critical_reviewer": "critical_reviewer",
            "synthesis": "synthesis",
            "END": END
        }
    )
    
    workflow.add_edge("literature_reviewer", "supervisor")
    workflow.add_edge("technical_analyzer", "supervisor")
    workflow.add_edge("critical_reviewer", "supervisor")
    workflow.add_edge("synthesis", "supervisor")
    
    logger.info("Graph edges (transitions) configured")
    
    compiled_workflow = workflow.compile()
    
    logger.success("Workflow compilation complete")
    logger.info("Architecture: Hierarchical Multi-Agent System")
    logger.info("Coordination: Supervisor-based routing")
    logger.info("Communication: Shared state (blackboard pattern)")
    
    return compiled_workflow


def visualize_workflow_structure():
    structure = """
    Multi-Agent Workflow Structure:
    ------------------------------------------------------------
    
    Entry Point: Supervisor (hierarchical coordinator)
    
    Possible Paths:
    1. Supervisor -> Literature Reviewer -> Supervisor
    2. Supervisor -> Technical Analyzer -> Supervisor
    3. Supervisor -> Critical Reviewer -> Supervisor
    4. Supervisor -> Synthesis -> Supervisor -> END
    """
    return structure


def run_workflow(workflow: StateGraph, initial_state: AgentState) -> AgentState:
    logger.header("Starting multi-agent workflow execution")
    logger.info(f"Input: {len(initial_state['paper_abstract'])} char paper abstract")
    logger.info(f"Target: Complete research paper review\n")
    
    try:
        final_state = workflow.invoke(initial_state)
        
        total_agents = len(final_state.get("messages", []))
        iterations = final_state.get("iteration_count", 0)
        
        logger.header("Workflow execution complete")
        logger.info(f"Total agent executions: {total_agents}")
        logger.info(f"Supervisor iterations: {iterations}")
        logger.info(f"Analysis complete: {final_state.get('analysis_complete', False)}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise


def display_workflow_summary(final_state: AgentState):
    logger.section("Agent contribution summary")
    
    contributions = []
    
    if final_state.get("literature_findings"):
        contributions.append(("Literature Reviewer", 
                            len(final_state["literature_findings"]), 
                            "Research context and key concepts"))
    
    if final_state.get("technical_analysis"):
        contributions.append(("Technical Analyzer", 
                            len(final_state["technical_analysis"]),
                            "Methodology evaluation"))
    
    if final_state.get("critical_review"):
        contributions.append(("Critical Reviewer", 
                            len(final_state["critical_review"]),
                            "Weakness identification and improvements"))
    
    if final_state.get("final_report"):
        contributions.append(("Synthesis Agent", 
                            len(final_state["final_report"]),
                            "Integrated final report"))
    
    for agent, chars, contribution in contributions:
        logger.info(f"{agent:30} | {chars:5} chars | {contribution}")
    
    total_output = sum(c[1] for c in contributions)
    logger.info(f"\n{'Total Output':30} | {total_output:5} chars")
    logger.info(f"Emergent Value: Comprehensive review from specialized analyses")
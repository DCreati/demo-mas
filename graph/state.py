from typing import TypedDict, List, Annotated
from typing_extensions import NotRequired


class AgentState(TypedDict):
    
    paper_abstract: str
    
    messages: List[dict]
    next_agent: str
    
    literature_findings: NotRequired[str]
    technical_analysis: NotRequired[str]
    critical_review: NotRequired[str]
    critical_evaluation: NotRequired[str]
    final_report: NotRequired[str]
    
    analysis_complete: bool
    iteration_count: NotRequired[int]
    needs_rerun: NotRequired[list]  # Agents to rerun: ["literature_reviewer"] or ["technical_analyzer"]
    rerun_count: NotRequired[dict]  # Track how many times each agent has been rerun
    literature_rerun_count: NotRequired[int]
    technical_rerun_count: NotRequired[int]


def create_initial_state(paper_abstract: str) -> AgentState:
    return AgentState(
        paper_abstract=paper_abstract,
        messages=[],
        next_agent="supervisor",
        analysis_complete=False,
        iteration_count=0,
        needs_rerun=[],
        rerun_count={},
        literature_rerun_count=0,
        technical_rerun_count=0
    )


def get_state_summary(state: AgentState) -> str:
    summary = []
    summary.append(f"Next Agent: {state.get('next_agent', 'None')}")
    summary.append(f"Messages: {len(state.get('messages', []))} agent communications")
    summary.append(f"Literature Findings: {'done' if state.get('literature_findings') else 'pending'}")
    summary.append(f"Technical Analysis: {'done' if state.get('technical_analysis') else 'pending'}")
    summary.append(f"Critical Review: {'done' if state.get('critical_review') else 'pending'}")
    summary.append(f"Final Report: {'done' if state.get('final_report') else 'pending'}")
    summary.append(f"Complete: {state.get('analysis_complete', False)}")
    summary.append(f"Iterations: {state.get('iteration_count', 0)}")
    
    return "\n".join(summary)


STATE_FIELD_DESCRIPTIONS = {
    "paper_abstract": "Input research paper abstract for analysis",
    "messages": "Agent communication log (blackboard pattern)",
    "next_agent": "Routing decision for hierarchical coordination",
    "literature_findings": "Literature review results",
    "technical_analysis": "Technical methodology analysis",
    "critical_review": "Critical evaluation and improvements",
    "critical_evaluation": "JSON output from critical reviewer with quality assessment and rerun decision",
    "final_report": "Synthesized final review",
    "analysis_complete": "Workflow termination flag",
    "iteration_count": "Safety counter to prevent infinite loops",
    "needs_rerun": "List of agent names to rerun based on critical reviewer assessment",
    "rerun_count": "Track how many times each agent has been rerun",
    "literature_rerun_count": "Counter for literature reviewer reruns",
    "technical_rerun_count": "Counter for technical analyzer reruns"
}
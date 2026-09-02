"""LangGraph agent workflow compiler."""
from typing import Dict, Any
from app.agents.state import AgentState
from app.agents.planner import plan_node
from app.agents.retriever import retrieve_node
from app.agents.responder import respond_node

class OmniFetchGraph:
    """Compiled Agent graph connecting Planner -> Retriever -> Responder."""

    def __init__(self):
        self._graph = None
        self._init_graph()

    def _init_graph(self):
        try:
            from langgraph.graph import StateGraph, END
            workflow = StateGraph(AgentState)
            workflow.add_node("planner", plan_node)
            workflow.add_node("retriever", retrieve_node)
            workflow.add_node("responder", respond_node)

            workflow.set_entry_point("planner")
            
            def check_safety(state: AgentState):
                return "responder" if not state.is_safe else "retriever"

            workflow.add_conditional_edges("planner", check_safety, {
                "responder": "responder",
                "retriever": "retriever"
            })
            workflow.add_edge("retriever", "responder")
            workflow.add_edge("responder", END)
            self._graph = workflow.compile()
        except Exception:
            # Native Python workflow runner fallback
            self._graph = None

    def query(self, question: str) -> AgentState:
        state = AgentState(query=question)
        if self._graph is not None:
            try:
                payload = state.model_dump() if hasattr(state, "model_dump") else state.dict()
                res_dict = self._graph.invoke(payload)
                if isinstance(res_dict, dict):
                    return AgentState(**res_dict)
                elif isinstance(res_dict, AgentState):
                    return res_dict
            except Exception:
                pass
            
        # Native sequential execution
        p_res = plan_node(state)
        for k, v in p_res.items():
            setattr(state, k, v)
            
        if not state.is_safe:
            return state
            
        r_res = retrieve_node(state)
        for k, v in r_res.items():
            setattr(state, k, v)
            
        resp_res = respond_node(state)
        for k, v in resp_res.items():
            setattr(state, k, v)
            
        return state

agent_graph = OmniFetchGraph()

from langgraph.graph import StateGraph, END, START

from app.graph.state import State
from app.graph.nodes import (
    classifier_node,
    extractor_node,
    compressor_node,
    critic_node,
    save_node,
)
from app.graph.edges import should_retry, increment_retry


def build_pipeline() -> StateGraph:
    """
    Assembles and compiles the ContextBridge extraction pipeline.

    Graph flow:
        START
          → classifier_node       (detect domain)
          → extractor_node        (extract 5-field brain)
          → compressor_node       (generate briefing)
          → critic_node           (score briefing)
          → [conditional edge]
              if score >= 7  → save_node → END
              if score < 7   → increment_retry → extractor_node (loop)
    """
    graph = StateGraph(State)

    graph.add_node("classifier_node",    classifier_node)
    graph.add_node("extractor_node",     extractor_node)
    graph.add_node("compressor_node",    compressor_node)
    graph.add_node("critic_node",        critic_node)
    graph.add_node("increment_retry",    increment_retry)
    graph.add_node("save_node",          save_node)

    graph.add_edge(START,              "classifier_node")
    graph.add_edge("classifier_node",  "extractor_node")
    graph.add_edge("extractor_node",   "compressor_node")
    graph.add_edge("compressor_node",  "critic_node")

    graph.add_conditional_edges(
        "critic_node",    
        should_retry,      
        {
            "save_node":       "save_node",      
            "extractor_node":  "increment_retry", 
        }
    )

    graph.add_edge("increment_retry", "extractor_node")

    graph.add_edge("save_node", END)

    return graph.compile()


pipeline = build_pipeline()
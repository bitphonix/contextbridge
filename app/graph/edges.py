from app.graph.state import State

# Passing threshold — critic score must meet or exceed this
PASS_SCORE = 7

# Maximum retries before we accept whatever we have
MAX_RETRIES = 2


def should_retry(state: State) -> str:
    """
    Conditional edge function called after critic_node.

    LangGraph calls this function and routes to whichever node name
    this function returns as a string.

    Returns:
        "extractor_node"  → retry the extraction with critic feedback
        "save_node"       → briefing passed, persist and finish
    """
    score      = state.get("critic_score", 0)
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        return "save_node"

    if score >= PASS_SCORE:
        return "save_node"

    return "extractor_node"


def increment_retry(state: State) -> dict:
    """
    Small helper node that increments retry_count before looping back
    to extractor_node. Keeps the counter logic out of extractor_node itself.
    """
    return {"retry_count": state.get("retry_count", 0) + 1}
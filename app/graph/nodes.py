import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from app.graph.state import State
from app.models.brain import BrainOutput, CriticOutput, ClassifierOutput

_llm_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)

_llm_pro = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)

_classifier_chain = _llm_flash.with_structured_output(ClassifierOutput)
_critic_chain     = _llm_flash.with_structured_output(CriticOutput)


def _parse_json_response(text: str) -> dict:
    """
    Strips markdown code fences from Gemini's response and parses JSON.
    Gemini often wraps JSON in ```json ... ``` even when asked not to.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def classifier_node(state: State) -> dict:
    prompt = f"""Analyse this AI conversation and classify its primary domain.

Conversation:
{state["raw_conversation"][:3000]}

Choose the single most appropriate domain from:
software_engineering, data_science, research, writing, design, business, general
"""
    try:
        result: ClassifierOutput = _classifier_chain.invoke(prompt)
        return {"domain": result.domain}
    except Exception as e:
        return {"domain": "general", "error": f"Classifier failed: {str(e)}"}


def extractor_node(state: State) -> dict:
    """
    Uses plain JSON prompt instead of .with_structured_output() to avoid
    Gemini's strict schema validation rejecting List[str] fields.
    """
    feedback_section = ""
    if state.get("critic_feedback"):
        feedback_section = f"""
PREVIOUS ATTEMPT FEEDBACK (you must address these issues):
{state["critic_feedback"]}
"""

    prompt = f"""You are extracting a structured "session brain" from an AI conversation.
This brain will allow someone to continue this work in a completely new chat with zero loss of context.

Domain: {state.get("domain", "general")}
{feedback_section}
Conversation to analyse:
{state["raw_conversation"]}

Return ONLY a valid JSON object with exactly these fields — no markdown, no explanation, just raw JSON:

{{
  "goal": "One sentence describing what the user was trying to accomplish. Be specific.",
  "decisions": ["Chose X over Y because Z", "..."],
  "dead_ends": ["Tried X — failed because Y", "..."],
  "current_state": "Exactly where things stand at the end of this conversation.",
  "next_steps": ["First immediate action", "Second action", "..."]
}}

Rules:
- decisions: specific choices locked in, with brief reasoning. At least one item.
- dead_ends: approaches tried and rejected. Empty list [] if none.
- next_steps: concrete, ordered, actionable. At least one item.
- Return raw JSON only — no ```json fences, no extra text.
"""
    try:
        response = _llm_pro.invoke(prompt)
        data = _parse_json_response(response.content)
        brain = BrainOutput(**data)
        return {
            "goal":          brain.goal,
            "decisions":     brain.decisions,
            "dead_ends":     brain.dead_ends,
            "current_state": brain.current_state,
            "next_steps":    brain.next_steps,
        }
    except Exception as e:
        return {"error": f"Extractor failed: {str(e)}"}


def compressor_node(state: State) -> dict:
    decisions_text  = "\n".join(f"  - {d}" for d in (state.get("decisions") or []))
    dead_ends_text  = "\n".join(f"  - {d}" for d in (state.get("dead_ends") or []))
    next_steps_text = "\n".join(f"  - {s}" for s in (state.get("next_steps") or []))

    prompt = f"""You are writing a context briefing that will be pasted at the start
of a new AI chat session. The reader is an AI model picking up this work cold.

Write a single, dense paragraph (150–250 words) that gives the AI everything
it needs to continue without asking clarifying questions.

Source brain:
GOAL: {state.get("goal")}

DECISIONS MADE:
{decisions_text}

DEAD ENDS (do NOT suggest these again):
{dead_ends_text}

CURRENT STATE: {state.get("current_state")}

NEXT STEPS:
{next_steps_text}

Rules:
- Write in second person ("We are building..." or "The project is...")
- Mention dead ends explicitly so the AI doesn't re-suggest them
- End with the single most important next action
- No headers, no bullet points — one flowing paragraph
"""
    try:
        response = _llm_pro.invoke(prompt)
        return {"briefing": response.content.strip()}
    except Exception as e:
        return {"error": f"Compressor failed: {str(e)}"}


def critic_node(state: State) -> dict:
    prompt = f"""You are evaluating a context briefing for an AI handoff.

Score this briefing on how well it would allow a NEW AI model to continue
the original work WITHOUT seeing the original conversation.

Original brain fields for reference:
- Goal: {state.get("goal")}
- Decisions: {state.get("decisions")}
- Dead ends: {state.get("dead_ends")}
- Current state: {state.get("current_state")}
- Next steps: {state.get("next_steps")}

Briefing to evaluate:
{state.get("briefing")}

Score 0–10. A passing score is 7 or above.
If failing, be specific about what is missing or unclear.
"""
    try:
        result: CriticOutput = _critic_chain.invoke(prompt)
        return {
            "critic_score":    result.score,
            "critic_feedback": result.feedback,
        }
    except Exception as e:
        return {"critic_score": 7, "critic_feedback": None,
                "error": f"Critic failed (defaulted pass): {str(e)}"}


def save_node(state: State) -> dict:
    from app.db.mongo import save_brain
    from datetime import datetime, timezone

    document = {
        "user_id":       state.get("user_id"),
        "domain":        state.get("domain"),
        "goal":          state.get("goal"),
        "decisions":     state.get("decisions"),
        "dead_ends":     state.get("dead_ends"),
        "current_state": state.get("current_state"),
        "next_steps":    state.get("next_steps"),
        "briefing":      state.get("briefing"),
        "critic_score":  state.get("critic_score"),
        "created_at":    datetime.now(timezone.utc),
    }

    try:
        brain_id = save_brain(document)
        return {"brain_id": brain_id}
    except Exception as e:
        return {"error": f"Save failed: {str(e)}"}
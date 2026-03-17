from pydantic import BaseModel, Field
from typing import Optional, List


class BrainOutput(BaseModel):
    """
    Structured output from the Extractor node.
    Gemini is instructed to return JSON matching this schema exactly.
    LangChain's .with_structured_output(BrainOutput) enforces it.
    """

    goal: str = Field(
        description=(
            "One sentence describing what the user was trying to accomplish "
            "in this conversation. Be specific — not 'coding help' but "
            "'building a LangGraph agent that monitors a GitHub repo'."
        )
    )

    decisions: List[str] = Field(
        description=(
            "Specific choices that were locked in during this conversation, "
            "each with a brief reason. Format: 'Chose X over Y because Z'. "
            "Only include decisions that would affect future work."
        )
    )

    dead_ends: List[str] = Field(
        description=(
            "Approaches that were explicitly tried and rejected or abandoned. "
            "Format: 'Tried X — failed because Y'. "
            "If nothing failed, return an empty list."
        ),
        default_factory=list,
    )

    current_state: str = Field(
        description=(
            "Exactly where things stand at the end of this conversation. "
            "What is working, what is broken, what is blocked. "
            "Be precise — a new model should know exactly what to do next."
        )
    )

    next_steps: List[str] = Field(
        description=(
            "The immediate actions that were about to happen when the "
            "conversation ended. Ordered by priority. "
            "Each step should be concrete and actionable."
        )
    )


class CriticOutput(BaseModel):
    """
    Structured output from the Critic node.
    Scores the briefing and explains what needs to improve.
    """

    score: int = Field(
        description=(
            "Quality score from 0 to 10. "
            "10 = a new model can continue the work perfectly with this briefing alone. "
            "0 = completely useless without the original conversation. "
            "Score >= 7 is passing."
        ),
        ge=0,
        le=10,
    )

    feedback: Optional[str] = Field(
        description=(
            "If score < 7, explain specifically what is missing or unclear. "
            "The Extractor will use this feedback to retry. "
            "If score >= 7, return null."
        ),
        default=None,
    )

    is_self_contained: bool = Field(
        description=(
            "True if someone reading only the briefing (not the original conversation) "
            "would have enough context to continue the work."
        )
    )


class ClassifierOutput(BaseModel):
    """
    Structured output from the Classifier node.
    """

    domain: str = Field(
        description=(
            "The primary domain of this conversation. "
            "Must be one of: software_engineering, data_science, research, "
            "writing, design, business, general."
        )
    )

    confidence: float = Field(
        description="Confidence in the domain classification, 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
"""
ContextBridge test suite.
Tests cover: Pydantic models, edge routing logic, and FastAPI endpoints.
We mock LLM calls so tests run without real API keys in CI.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Model tests ───────────────────────────────────────────────────────────────

class TestBrainOutput:
    def test_valid_brain_output(self):
        from app.models.brain import BrainOutput
        brain = BrainOutput(
            goal="Build a LangGraph agent for context portability",
            decisions=["Chose MongoDB over PostgreSQL because brain docs are JSON-shaped"],
            dead_ends=["Tried with_structured_output — failed due to Gemini schema bug"],
            current_state="Pipeline is working end to end, frontend is live",
            next_steps=["Deploy to DigitalOcean", "Write README"],
        )
        assert brain.goal.startswith("Build")
        assert len(brain.decisions) == 1
        assert len(brain.dead_ends) == 1
        assert len(brain.next_steps) == 2

    def test_brain_output_empty_dead_ends(self):
        from app.models.brain import BrainOutput
        brain = BrainOutput(
            goal="Test goal",
            decisions=["Decision one"],
            dead_ends=[],
            current_state="Current state text",
            next_steps=["Next step one"],
        )
        assert brain.dead_ends == []

    def test_brain_output_missing_required_field(self):
        from app.models.brain import BrainOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BrainOutput(
                decisions=["A decision"],
                dead_ends=[],
                current_state="state",
                next_steps=["step"],
                # goal is missing — should raise
            )


class TestCriticOutput:
    def test_valid_critic_output_passing(self):
        from app.models.brain import CriticOutput
        critic = CriticOutput(score=9, feedback=None, is_self_contained=True)
        assert critic.score == 9
        assert critic.is_self_contained is True

    def test_valid_critic_output_failing(self):
        from app.models.brain import CriticOutput
        critic = CriticOutput(
            score=4,
            feedback="Missing specific decisions and dead ends",
            is_self_contained=False
        )
        assert critic.score == 4
        assert "decisions" in critic.feedback

    def test_critic_score_out_of_range(self):
        from app.models.brain import CriticOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CriticOutput(score=11, feedback=None, is_self_contained=True)

    def test_critic_score_negative(self):
        from app.models.brain import CriticOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CriticOutput(score=-1, feedback=None, is_self_contained=True)


class TestClassifierOutput:
    def test_valid_classifier_output(self):
        from app.models.brain import ClassifierOutput
        clf = ClassifierOutput(domain="software_engineering", confidence=0.95)
        assert clf.domain == "software_engineering"
        assert clf.confidence == 0.95

    def test_confidence_out_of_range(self):
        from app.models.brain import ClassifierOutput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClassifierOutput(domain="research", confidence=1.5)


# ── Edge routing tests ────────────────────────────────────────────────────────

class TestEdgeRouting:
    def test_passes_on_high_score(self):
        from app.graph.edges import should_retry
        state = {"critic_score": 8, "retry_count": 0}
        assert should_retry(state) == "save_node"

    def test_passes_on_exact_threshold(self):
        from app.graph.edges import should_retry
        state = {"critic_score": 7, "retry_count": 0}
        assert should_retry(state) == "save_node"

    def test_retries_on_low_score(self):
        from app.graph.edges import should_retry
        state = {"critic_score": 5, "retry_count": 0}
        assert should_retry(state) == "extractor_node"

    def test_force_passes_at_max_retries(self):
        from app.graph.edges import should_retry
        # Even with a failing score, max retries forces a pass
        state = {"critic_score": 3, "retry_count": 2}
        assert should_retry(state) == "save_node"

    def test_retries_before_max(self):
        from app.graph.edges import should_retry
        state = {"critic_score": 4, "retry_count": 1}
        assert should_retry(state) == "extractor_node"

    def test_increment_retry(self):
        from app.graph.edges import increment_retry
        state = {"retry_count": 1}
        result = increment_retry(state)
        assert result["retry_count"] == 2

    def test_increment_retry_from_zero(self):
        from app.graph.edges import increment_retry
        state = {"retry_count": 0}
        result = increment_retry(state)
        assert result["retry_count"] == 1


# ── State tests ───────────────────────────────────────────────────────────────

class TestState:
    def test_state_fields_exist(self):
        from app.graph.state import State
        # TypedDict keys should include all expected fields
        keys = State.__annotations__.keys()
        assert "raw_conversation" in keys
        assert "goal" in keys
        assert "decisions" in keys
        assert "dead_ends" in keys
        assert "current_state" in keys
        assert "next_steps" in keys
        assert "briefing" in keys
        assert "critic_score" in keys
        assert "retry_count" in keys
        assert "brain_id" in keys


# ── API endpoint tests ────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Creates a FastAPI test client with the pipeline mocked out.
    This lets us test API logic without making real Gemini or MongoDB calls.
    """
    mock_result = {
        "brain_id":      "507f1f77bcf86cd799439011",
        "domain":        "software_engineering",
        "goal":          "Build a scalable event pipeline",
        "decisions":     ["Chose ClickHouse over PostgreSQL for analytics workload"],
        "dead_ends":     ["Rejected Kafka due to team complexity"],
        "current_state": "Architecture decided, not yet implemented",
        "next_steps":    ["Set up ClickHouse instance"],
        "briefing":      "We are building a SaaS analytics platform...",
        "critic_score":  9,
        "retry_count":   0,
        "error":         None,
    }

    with patch("app.graph.pipeline.pipeline") as mock_pipeline:
        mock_pipeline.invoke.return_value = mock_result
        # Re-import app after patching
        import importlib
        import app.main as main_module
        importlib.reload(main_module)
        from app.main import app as fastapi_app
        yield TestClient(fastapi_app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        # Health doesn't need mocking — no external calls
        import os
        os.environ.setdefault("GEMINI_API_KEY", "test")
        os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/test")
        os.environ.setdefault("SENTRY_DSN", "")
        with patch("sentry_sdk.init"):
            with patch("app.graph.pipeline.build_pipeline"):
                from fastapi.testclient import TestClient
                from app.main import app
                c = TestClient(app)
                response = c.get("/health")
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}


class TestExtractEndpoint:
    def test_rejects_short_conversation(self):
        import os
        os.environ.setdefault("GEMINI_API_KEY", "test")
        os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/test")
        os.environ.setdefault("SENTRY_DSN", "")
        with patch("sentry_sdk.init"):
            with patch("app.graph.pipeline.build_pipeline"):
                from fastapi.testclient import TestClient
                from app.main import app
                c = TestClient(app)
                response = c.post("/extract", json={"conversation": "too short"})
                assert response.status_code == 400
                assert "too short" in response.json()["detail"]
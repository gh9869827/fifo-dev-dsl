"""Tests for LLMRuntimeContext with LlmBackend integration."""
import warnings
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.common.llm_abstraction import LlmRequest

# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

class MockBackend:
    """Mock LLM backend for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_request = None

    def complete(self, req: LlmRequest) -> str:
        """Mock complete method."""
        self.call_count += 1
        self.last_request = req
        return "MOCK_RESPONSE"


class TestLLMRuntimeContextBackend:
    """Tests for LLMRuntimeContext with llm_backend parameter."""

    def test_llm_backend_parameter(self):
        """Test that llm_backend parameter is accepted and used."""
        mock_backend = MockBackend()

        ctx = LLMRuntimeContext(
            tools=[],
            query_sources=[],
            llm_backend_dsl=mock_backend, # type: ignore
            llm_backend_reasoning=mock_backend # type: ignore
        )

        # Call the LLM
        result = ctx.call_llm_dsl("system prompt", "user prompt")

        # Verify it was called
        assert result == "MOCK_RESPONSE"
        assert mock_backend.call_count == 1
        assert mock_backend.last_request is not None
        assert mock_backend.last_request.system_prompt == "system prompt"
        assert mock_backend.last_request.user_prompt == "user prompt"
        assert mock_backend.last_request.max_new_tokens == 1024
        assert mock_backend.last_request.temperature == 0.0

    def test_deprecated_parameters_emit_warning(self):
        """Test that using deprecated parameters emits a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = LLMRuntimeContext(
                tools=[],
                query_sources=[],
                container_name="test-container"
            )

            # Verify warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_deprecated_parameters_create_airlock_backend(self):
        """Test that deprecated parameters create an AirlockBackend."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            ctx = LLMRuntimeContext(
                tools=[],
                query_sources=[],
                container_name="test-container",
                intent_sequencer_adapter="test-adapter",
                host="http://test:8000"
            )

            # Verify the backends were created (we cannot easily test it is an AirlockBackend
            # without importing airlock dependencies, but we can test it exists)
            assert ctx._llm_backend_dsl is not None
            assert ctx._llm_backend_reasoning is not None

    def test_llm_backend_takes_precedence(self):
        """Test that llm_backend parameter takes precedence over deprecated params."""
        mock_backend = MockBackend()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            ctx = LLMRuntimeContext(
                tools=[],
                query_sources=[],
                llm_backend_dsl=mock_backend, # type: ignore
                llm_backend_reasoning=mock_backend, # type: ignore
                container_name="test-container"  # Should be ignored
            )

            # Warning should still be emitted
            assert len(w) == 1

        # But the mock backend should be used
        result = ctx.call_llm_dsl("test", "test")
        assert result == "MOCK_RESPONSE"
        assert mock_backend.call_count == 1

    def test_no_warning_when_only_llm_backend_provided(self):
        """Test that no warning is emitted when only llm_backend is provided."""
        mock_backend = MockBackend()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = LLMRuntimeContext(
                tools=[],
                query_sources=[],
                llm_backend_dsl=mock_backend, # type: ignore
                llm_backend_reasoning=mock_backend # type: ignore
            )

            # No warning should be emitted
            assert len(w) == 0

    def test_property_access_emits_warnings(self):
        """Test that accessing deprecated properties emits warnings."""
        mock_backend = MockBackend()

        ctx = LLMRuntimeContext(
            tools=[],
            query_sources=[],
            llm_backend_dsl=mock_backend, # type: ignore
            llm_backend_reasoning=mock_backend, # type: ignore
        )

        # Test each deprecated property
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ctx.container_name
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ctx.base_model
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ctx.host
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ctx.intent_sequencer_adapter
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_backward_compatibility_default_values(self):
        """Test that default values work for backward compatibility."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            # Create context with no parameters (using defaults)
            ctx = LLMRuntimeContext(
                tools=[],
                query_sources=[]
            )

            # Should have the llm_backends created
            assert ctx._llm_backend_dsl
            assert ctx._llm_backend_reasoning is not None

"""Tests for LlmRequest and LlmBackend implementations."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from fifo_dev_dsl.common.llm_abstraction import LlmRequest, OpenAICompatibleBackend


class TestLlmRequest:
    """Tests for LlmRequest dataclass."""

    def test_llm_request_default_values(self):
        """Test that LlmRequest has correct default values."""
        req = LlmRequest(
            system_prompt="system prompt",
            user_prompt="user prompt"
        )
        
        assert req.system_prompt == "system prompt"
        assert req.user_prompt == "user prompt"
        assert req.max_new_tokens == 1024
        assert req.temperature == 0.0
        assert req.reasoning_level == "low"

    def test_llm_request_custom_values(self):
        """Test that LlmRequest accepts custom values."""
        req = LlmRequest(
            system_prompt="custom system",
            user_prompt="custom user",
            max_new_tokens=512,
            temperature=0.7,
            reasoning_level="high"
        )
        
        assert req.system_prompt == "custom system"
        assert req.user_prompt == "custom user"
        assert req.max_new_tokens == 512
        assert req.temperature == 0.7
        assert req.reasoning_level == "high"

    def test_llm_request_reasoning_level_medium(self):
        """Test that reasoning_level can be set to medium."""
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user",
            reasoning_level="medium"
        )
        
        assert req.reasoning_level == "medium"

    def test_llm_request_frozen(self):
        """Test that LlmRequest is frozen (immutable)."""
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
            req.reasoning_level = "high"


class TestOpenAICompatibleBackend:
    """Tests for OpenAICompatibleBackend."""

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend.__init__', return_value=None)
    def test_openai_backend_passes_reasoning_level(self, mock_init):
        """Test that OpenAICompatibleBackend passes reasoning_level to API."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "TEST_OUTPUT"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create backend instance manually and inject the mock client
        backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
        backend._model = "test-model"
        backend._client = mock_client
        
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user",
            reasoning_level="high"
        )
        
        result = backend.complete(req)
        
        # Verify the API was called with reasoning_level
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['reasoning_level'] == "high"
        assert result == "TEST_OUTPUT"

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend.__init__', return_value=None)
    def test_openai_backend_default_reasoning_level(self, mock_init):
        """Test that default reasoning_level is passed correctly."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "TEST_OUTPUT"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create backend instance manually and inject the mock client
        backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
        backend._model = "test-model"
        backend._client = mock_client
        
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )
        
        result = backend.complete(req)
        
        # Verify the API was called with default reasoning_level
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['reasoning_level'] == "low"
        assert result == "TEST_OUTPUT"

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend.__init__', return_value=None)
    def test_openai_backend_passes_all_parameters(self, mock_init):
        """Test that all LlmRequest parameters are passed to API."""
        # Create a mock client
        mock_client = MagicMock()
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "DSL CODE"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create backend instance manually and inject the mock client
        backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
        backend._model = "test-model"
        backend._client = mock_client
        
        req = LlmRequest(
            system_prompt="You are a DSL generator",
            user_prompt="Convert this to DSL",
            max_new_tokens=2048,
            temperature=0.5,
            reasoning_level="medium"
        )
        
        result = backend.complete(req)
        
        # Verify all parameters are passed correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        
        assert call_kwargs['model'] == "test-model"
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 2048
        assert call_kwargs['reasoning_level'] == "medium"
        assert len(call_kwargs['messages']) == 2
        assert call_kwargs['messages'][0]['role'] == "system"
        assert call_kwargs['messages'][0]['content'] == "You are a DSL generator"
        assert call_kwargs['messages'][1]['role'] == "user"
        assert call_kwargs['messages'][1]['content'] == "Convert this to DSL"
        assert result == "DSL CODE"

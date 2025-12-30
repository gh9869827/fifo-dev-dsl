"""Tests for LlmRequest and LlmBackend implementations."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from fifo_dev_dsl.common.llm_abstraction import LlmRequest, OpenAICompatibleBackend, AirlockBackend


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
        assert req.reasoning_effort is None

    def test_llm_request_custom_values(self):
        """Test that LlmRequest accepts custom values."""
        req = LlmRequest(
            system_prompt="custom system",
            user_prompt="custom user",
            max_new_tokens=512,
            temperature=0.7,
            reasoning_effort="high"
        )
        
        assert req.system_prompt == "custom system"
        assert req.user_prompt == "custom user"
        assert req.max_new_tokens == 512
        assert req.temperature == 0.7
        assert req.reasoning_effort == "high"

    def test_llm_request_reasoning_effort_medium(self):
        """Test that reasoning_effort can be set to medium."""
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user",
            reasoning_effort="medium"
        )
        
        assert req.reasoning_effort == "medium"

    def test_llm_request_frozen(self):
        """Test that LlmRequest is frozen (immutable)."""
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
            req.reasoning_effort = "high"


class TestOpenAICompatibleBackend:
    """Tests for OpenAICompatibleBackend."""

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend.__init__', return_value=None)
    def test_openai_backend_passes_reasoning_effort(self, mock_init):
        """Test that OpenAICompatibleBackend passes reasoning_effort to API when set."""
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
            reasoning_effort="high"
        )
        
        result = backend.complete(req)
        
        # Verify the API was called with reasoning_effort
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['reasoning_effort'] == "high"
        assert result == "TEST_OUTPUT"

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend.__init__', return_value=None)
    def test_openai_backend_omits_reasoning_effort_when_none(self, mock_init):
        """Test that reasoning_effort is not passed when None."""
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
        
        # Verify the API was called without reasoning_effort
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert 'reasoning_effort' not in call_kwargs
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
            reasoning_effort="medium"
        )
        
        result = backend.complete(req)
        
        # Verify all parameters are passed correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        
        assert call_kwargs['model'] == "test-model"
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 2048
        assert call_kwargs['reasoning_effort'] == "medium"
        assert len(call_kwargs['messages']) == 2
        assert call_kwargs['messages'][0]['role'] == "system"
        assert call_kwargs['messages'][0]['content'] == "You are a DSL generator"
        assert call_kwargs['messages'][1]['role'] == "user"
        assert call_kwargs['messages'][1]['content'] == "Convert this to DSL"
        assert result == "DSL CODE"


class TestAirlockBackend:
    """Tests for AirlockBackend."""

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend.__init__', return_value=None)
    def test_airlock_backend_default_model(self, mock_init):
        """Test that AirlockBackend defaults to Phi4MiniInstruct when no model is provided."""
        # Create mock classes
        mock_model_enum = MagicMock()
        mock_model_enum.Phi4MiniInstruct = "Phi4MiniInstruct"
        
        mock_message = MagicMock()
        mock_message.system = MagicMock(return_value="system_message")
        mock_message.user = MagicMock(return_value="user_message")
        
        mock_gen_params = MagicMock()
        mock_call_server = MagicMock(return_value="DSL_OUTPUT")
        
        # Create backend instance manually
        backend = AirlockBackend.__new__(AirlockBackend)
        backend._model_enum = mock_model_enum
        backend._message_cls = mock_message
        backend._generation_parameters_cls = mock_gen_params
        backend._call_airlock_model_server = mock_call_server
        backend._model = mock_model_enum.Phi4MiniInstruct
        backend._adapter = "test-adapter"
        backend._container_name = "test-container"
        backend._host = "http://localhost:8000"
        
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )
        
        result = backend.complete(req)
        
        # Verify the model server was called with the default model
        mock_call_server.assert_called_once()
        call_kwargs = mock_call_server.call_args.kwargs
        assert call_kwargs['model'] == mock_model_enum.Phi4MiniInstruct
        assert result == "DSL_OUTPUT"

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend.__init__', return_value=None)
    def test_airlock_backend_custom_model(self, mock_init):
        """Test that AirlockBackend uses the provided model."""
        # Create mock classes
        mock_model_enum = MagicMock()
        mock_model_enum.Phi4MultimodalInstruct = "Phi4MultimodalInstruct"
        
        mock_message = MagicMock()
        mock_message.system = MagicMock(return_value="system_message")
        mock_message.user = MagicMock(return_value="user_message")
        
        mock_gen_params = MagicMock()
        mock_call_server = MagicMock(return_value="DSL_OUTPUT")
        
        # Create backend instance manually
        backend = AirlockBackend.__new__(AirlockBackend)
        backend._model_enum = mock_model_enum
        backend._message_cls = mock_message
        backend._generation_parameters_cls = mock_gen_params
        backend._call_airlock_model_server = mock_call_server
        backend._model = mock_model_enum.Phi4MultimodalInstruct
        backend._adapter = "test-adapter"
        backend._container_name = "test-container"
        backend._host = "http://localhost:8000"
        
        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )
        
        result = backend.complete(req)
        
        # Verify the model server was called with the custom model
        mock_call_server.assert_called_once()
        call_kwargs = mock_call_server.call_args.kwargs
        assert call_kwargs['model'] == mock_model_enum.Phi4MultimodalInstruct
        assert result == "DSL_OUTPUT"

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend.__init__', return_value=None)
    def test_airlock_backend_passes_all_parameters(self, mock_init):
        """Test that AirlockBackend passes all parameters correctly."""
        # Create mock classes
        mock_model_enum = MagicMock()
        mock_model_enum.Phi4MiniInstruct = "Phi4MiniInstruct"
        
        mock_message = MagicMock()
        mock_message.system = MagicMock(return_value="system_message")
        mock_message.user = MagicMock(return_value="user_message")
        
        mock_gen_params = MagicMock(return_value="gen_params_instance")
        mock_call_server = MagicMock(return_value="DSL_OUTPUT")
        
        # Create backend instance manually
        backend = AirlockBackend.__new__(AirlockBackend)
        backend._model_enum = mock_model_enum
        backend._message_cls = mock_message
        backend._generation_parameters_cls = mock_gen_params
        backend._call_airlock_model_server = mock_call_server
        backend._model = mock_model_enum.Phi4MiniInstruct
        backend._adapter = "test-adapter"
        backend._container_name = "test-container"
        backend._host = "http://localhost:8000"
        
        req = LlmRequest(
            system_prompt="system prompt",
            user_prompt="user prompt",
            max_new_tokens=2048,
            temperature=0.5
        )
        
        result = backend.complete(req)
        
        # Verify all parameters are passed
        mock_call_server.assert_called_once()
        call_kwargs = mock_call_server.call_args.kwargs
        
        assert call_kwargs['model'] == mock_model_enum.Phi4MiniInstruct
        assert call_kwargs['adapter'] == "test-adapter"
        assert call_kwargs['container_name'] == "test-container"
        assert call_kwargs['host'] == "http://localhost:8000"
        assert len(call_kwargs['messages']) == 2
        
        # Verify generation parameters
        mock_gen_params.assert_called_once_with(
            max_new_tokens=2048,
            do_sample=True  # temperature > 0
        )
        assert call_kwargs['parameters'] == "gen_params_instance"
        assert result == "DSL_OUTPUT"

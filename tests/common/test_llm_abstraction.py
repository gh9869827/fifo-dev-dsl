"""Tests for LlmRequest and LlmBackend implementations."""
import argparse
from unittest.mock import MagicMock, patch, call
import pytest
from fifo_tool_airlock_model_env.common.models import Model
from fifo_dev_dsl.common.llm_abstraction import (
    LlmRequest,
    OpenAICompatibleBackend,
    AirlockBackend,
    _parse_backend_args,
    LlmBackendType,
    parse_cli_and_create_backends
)

# pyright: reportPrivateUsage=false
# pylint: disable=protected-access

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
            req.reasoning_effort = "high" # type: ignore


class TestOpenAICompatibleBackend:
    """Tests for OpenAICompatibleBackend."""

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend.__init__', return_value=None)
    def test_openai_backend_passes_reasoning_effort(self, _mock_init: MagicMock):
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
    def test_openai_backend_omits_reasoning_effort_when_none(self, _mock_init: MagicMock):
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
    def test_openai_backend_passes_all_parameters(self, _mock_init: MagicMock):
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

    def test_airlock_backend_default_model(self):
        """Test that AirlockBackend defaults to Phi4MiniInstruct when no model is provided."""

        # Create backend instance
        backend = AirlockBackend(
            container_name="test-container",
            adapter="test-adapter",
            host="http://localhost:8000",
            base_model=Model.Phi4MiniInstruct
        )

        # Override call to airlock model server
        mock_call_server = MagicMock(return_value="DSL_OUTPUT")
        backend._call_airlock_model_server = mock_call_server

        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )

        result = backend.complete(req)

        # Verify the model server was called with the default model
        mock_call_server.assert_called_once()
        call_kwargs = mock_call_server.call_args.kwargs
        assert call_kwargs['model'] == Model.Phi4MiniInstruct
        assert result == "DSL_OUTPUT"

    def test_airlock_backend_custom_model(self):
        """Test that AirlockBackend uses the provided model."""

        # Create backend instance
        backend = AirlockBackend(
            container_name="test-container",
            adapter="test-adapter",
            host="http://localhost:8000",
            base_model=Model.Phi4MultimodalInstruct
        )

        # Override call to airlock model server
        mock_call_server = MagicMock(return_value="DSL_OUTPUT")
        backend._call_airlock_model_server = mock_call_server

        req = LlmRequest(
            system_prompt="system",
            user_prompt="user"
        )

        result = backend.complete(req)

        # Verify the model server was called with the custom model
        mock_call_server.assert_called_once()
        call_kwargs = mock_call_server.call_args.kwargs
        assert call_kwargs['model'] == Model.Phi4MultimodalInstruct
        assert result == "DSL_OUTPUT"

    def test_airlock_backend_passes_all_parameters(self):
        """Test that AirlockBackend passes all parameters correctly."""

        # Create backend instance
        backend = AirlockBackend(
            container_name="test-container",
            adapter="test-adapter",
            host="http://localhost:8000",
            base_model=Model.Phi4MiniInstruct
        )

        # Override call to airlock model server
        mock_call_server = MagicMock(return_value="DSL_OUTPUT")
        backend._call_airlock_model_server = mock_call_server

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

        assert call_kwargs['model'] == Model.Phi4MiniInstruct
        assert call_kwargs['adapter'] == "test-adapter"
        assert call_kwargs['container_name'] == "test-container"
        assert call_kwargs['host'] == "http://localhost:8000"
        assert len(call_kwargs['messages']) == 2
        assert call_kwargs['parameters'].max_new_tokens == 2048
        assert call_kwargs['parameters'].do_sample is True
        assert result == "DSL_OUTPUT"


class TestAddBackendCliArguments:
    """Tests for add_backend_cli_arguments function."""

    def test_add_backend_cli_arguments_default_adapter_airlock(self):
        """Test that add_backend_cli_arguments adds all arguments with correct defaults."""
        global_parser = argparse.ArgumentParser()

        parsed_backends = _parse_backend_args(
            ["dsl=airlock"],
            default_adapter="test-adapter",
            prog=global_parser.prog,
            require_reasoning=False,
        )

        dsl = parsed_backends.dsl_args

        assert parsed_backends.dsl_backend_type == LlmBackendType.AIRLOCK
        assert dsl.host == "http://127.0.0.1:8000"
        assert dsl.container == "phi"
        assert dsl.model == "Phi4MiniInstruct"
        assert dsl.adapter == "test-adapter"

    def test_add_backend_cli_arguments_default_adapter_openaicompatible(self):
        """Test that add_backend_cli_arguments adds all arguments with correct defaults."""
        global_parser = argparse.ArgumentParser()

        parsed_backends = _parse_backend_args(
            [
                "dsl=openai-compatible",
                "--base-url",
                "http://127.0.0.1:8000"
            ],
            default_adapter="test-adapter",
            prog=global_parser.prog,
            require_reasoning=False,
        )

        dsl = parsed_backends.dsl_args

        assert parsed_backends.dsl_backend_type == LlmBackendType.OPENAI_COMPATIBLE
        assert dsl.base_url == "http://127.0.0.1:8000"
        assert dsl.adapter == "test-adapter"

    def test_add_backend_cli_arguments_accepts_custom_values_airlock_openaicompatible(self):
        """Test that custom values can be provided for all arguments."""
        global_parser = argparse.ArgumentParser()

        parsed_backends = _parse_backend_args(
            [
                "dsl=airlock",
                "--adapter", "test-adapter",
                "--container", "test-container",
                "--host", "http://127.0.0.1:8000/test-url1",
                "--model", "Phi4MiniMultimodal",

                "reasoning=openai-compatible",
                "--model", "test-model2",
                "--base-url", "http://127.0.0.1:8000/test-url2",
                "--api-key", "dummy-api-key",
            ],
            default_adapter="test-default-adapter",
            prog=global_parser.prog,
            require_reasoning=False,
        )

        dsl = parsed_backends.dsl_args
        reasoning = parsed_backends.reasoning_args

        assert parsed_backends.dsl_backend_type == LlmBackendType.AIRLOCK
        assert dsl.host == "http://127.0.0.1:8000/test-url1"
        assert dsl.container == "test-container"
        assert dsl.model == "Phi4MiniMultimodal"
        assert dsl.adapter == "test-adapter"

        assert reasoning is not None
        assert parsed_backends.reasoning_backend_type == LlmBackendType.OPENAI_COMPATIBLE
        assert reasoning.model == "test-model2"
        assert reasoning.base_url == "http://127.0.0.1:8000/test-url2"
        assert reasoning.api_key == "dummy-api-key"

    def test_add_backend_cli_arguments_accepts_custom_values_openaicompatible_airlock(self):
        """Test that custom values can be provided for all arguments."""
        global_parser = argparse.ArgumentParser()

        parsed_backends = _parse_backend_args(
            [
                "dsl=openai-compatible",
                "--adapter", "test-adapter",
                "--base-url", "http://127.0.0.1:8000/test-url1",
                "--api-key", "dummy-api-key",

                "reasoning=airlock",
                "--container", "test-container",
                "--host", "http://127.0.0.1:8000/test-url2",
                "--model", "Phi4MiniMultimodal",
            ],
            default_adapter="test-default-adapter",
            prog=global_parser.prog,
            require_reasoning=True,
        )

        dsl = parsed_backends.dsl_args
        reasoning = parsed_backends.reasoning_args

        assert parsed_backends.dsl_backend_type == LlmBackendType.OPENAI_COMPATIBLE
        assert dsl.adapter == "test-adapter"
        assert dsl.base_url == "http://127.0.0.1:8000/test-url1"
        assert dsl.api_key == "dummy-api-key"

        assert reasoning is not None
        assert parsed_backends.reasoning_backend_type == LlmBackendType.AIRLOCK
        assert reasoning.container == "test-container"
        assert reasoning.host == "http://127.0.0.1:8000/test-url2"
        assert reasoning.model == "Phi4MiniMultimodal"

    def test_add_backend_cli_arguments_backend_type_choices_1(self):
        """Test that backend-type only accepts valid choices."""
        global_parser = argparse.ArgumentParser()

        # Test invalid backend type
        with pytest.raises(SystemExit, match="invalid backend spec 'dsl=error'"):
            _parsed_backends = _parse_backend_args(
                [
                    "dsl=error",
                    "--adapter", "test-adapter",
                    "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key",

                    "reasoning=error",
                    "--container", "test-container",
                    "--host", "http://127.0.0.1:8000/test-url2",
                    "--model", "Phi4MiniMultimodal",
                ],
                default_adapter="test-default-adapter",
                prog=global_parser.prog,
                require_reasoning=True,
            )

    def test_add_backend_cli_arguments_backend_type_choices_2(self):
        """Test that backend-type only accepts valid choices."""
        global_parser = argparse.ArgumentParser()

        # Test invalid backend type
        with pytest.raises(SystemExit, match="invalid backend spec 'reasoning=error'"):
            _parsed_backends = _parse_backend_args(
                [
                    "dsl=openai-compatible",
                    "--adapter", "test-adapter",
                    "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key",

                    "reasoning=error",
                    "--container", "test-container",
                    "--host", "http://127.0.0.1:8000/test-url2",
                    "--model", "Phi4MiniMultimodal",
                ],
                default_adapter="test-default-adapter",
                prog=global_parser.prog,
                require_reasoning=True,
            )

    def test_add_backend_cli_arguments_missing_reasoning(self):
        """Test that reasoning is present when required."""
        global_parser = argparse.ArgumentParser()

        # Test invalid backend type
        with pytest.raises(SystemExit, match="missing required 'reasoning=...'"):
            _parsed_backends = _parse_backend_args(
                [
                    "dsl=openai-compatible",
                    "--adapter", "test-adapter",
                    "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key",
                ],
                default_adapter="test-default-adapter",
                prog=global_parser.prog,
                require_reasoning=True,
            )


class TestCreateBackendFromArgs:
    """Tests for create_backend_from_args function."""

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend')
    def test_create_airlock_backend_with_default_values(self, mock_airlock_backend: MagicMock):
        """Test creating AirlockBackend with default values - dsl only."""

        _ = parse_cli_and_create_backends(
            [
                "dsl=airlock"
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=False,
        )

        mock_airlock_backend.assert_called_once_with(
            container_name="phi",
            adapter="test-default-adapter",
            host="http://127.0.0.1:8000",
            base_model="Phi4MiniInstruct"
        )

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend')
    def test_create_airlock_backend_with_custom_values(self, mock_airlock_backend: MagicMock):
        """Test creating AirlockBackend with custom values - dsl only."""
        _ = parse_cli_and_create_backends(
            [
                "dsl=airlock",
                "--adapter", "test-adapter",
                "--container", "test-container",
                "--host", "http://127.0.0.1:8000/test-url",
                "--model", "Phi4MiniMultimodal",
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=False,
        )

        mock_airlock_backend.assert_called_once_with(
            container_name="test-container",
            adapter="test-adapter",
            host="http://127.0.0.1:8000/test-url",
            base_model="Phi4MiniMultimodal"
        )

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend')
    def test_create_airlock_backend_with_default_values_reasoning(self, mock_airlock_backend: MagicMock):
        """Test creating AirlockBackend with default values - dsl + reasoning only."""

        _ = parse_cli_and_create_backends(
            [
                "dsl=airlock",
                "reasoning=airlock"
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=True,
        )

        mock_airlock_backend.assert_has_calls(
            calls=[
                call(
                    container_name="phi",
                    adapter="test-default-adapter",
                    host="http://127.0.0.1:8000",
                    base_model="Phi4MiniInstruct"
                ),
                call(
                    container_name="phi",
                    adapter=None,
                    host="http://127.0.0.1:8000",
                    base_model="Phi4MiniInstruct"
                )

            ]
        )

    @patch('fifo_dev_dsl.common.llm_abstraction.AirlockBackend')
    def test_create_airlock_backend_with_custom_values_reasoning(self, mock_airlock_backend: MagicMock):
        """Test creating AirlockBackend with custom values - dsl + reasoning only."""
        _ = parse_cli_and_create_backends(
            [
                "dsl=airlock",
                "--adapter", "test-adapter1",
                "--container", "test-container1",
                "--host", "http://127.0.0.1:8000/test-url1",
                "--model", "Phi4MiniMultimodal",

                "reasoning=airlock",
                "--container", "test-container2",
                "--host", "http://127.0.0.1:8000/test-url2",
                "--model", "Phi4MiniMultimodal",
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=True,
        )

        mock_airlock_backend.assert_has_calls(
            calls=[
                call(
                    container_name="test-container1",
                    adapter="test-adapter1",
                    host="http://127.0.0.1:8000/test-url1",
                    base_model="Phi4MiniMultimodal"
                ),
                call(
                    container_name="test-container2",
                    adapter=None,
                    host="http://127.0.0.1:8000/test-url2",
                    base_model="Phi4MiniMultimodal"
                )

            ]
        )



    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend')
    def test_create_openaicompatible_backend_with_default_values(self, mock_airlock_backend: MagicMock):
        """Test creating OpenAICompatibleBackend with default values - dsl only."""

        _ = parse_cli_and_create_backends(
            [
                "dsl=openai-compatible",
                "--base-url", "http://127.0.0.1:8000/test-url1",
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=False,
        )

        mock_airlock_backend.assert_called_once_with(
            base_url="http://127.0.0.1:8000/test-url1",
            model="test-default-adapter",
            api_key="EMPTY"
        )

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend')
    def test_create_openaicompatible_backend_with_custom_values(self, mock_airlock_backend: MagicMock):
        """Test creating OpenAICompatibleBackend with custom values - dsl only."""

        _ = parse_cli_and_create_backends(
            [
                "dsl=openai-compatible",
                "--adapter", "test-adapter",
                "--base-url", "http://127.0.0.1:8000/test-url1",
                "--api-key", "dummy-api-key",
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=False,
        )

        mock_airlock_backend.assert_called_once_with(
            base_url="http://127.0.0.1:8000/test-url1",
            model="test-adapter",
            api_key="dummy-api-key"
        )

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend')
    def test_create_openaicompatible_backend_with_default_values_reasoning(self, mock_airlock_backend: MagicMock):
        """Test creating OpenAICompatibleBackend with default values - dsl + reasoning only."""

        _ = parse_cli_and_create_backends(
            [
                "dsl=openai-compatible",
                "--base-url", "http://127.0.0.1:8000/test-url1",

                "reasoning=openai-compatible",
                "--base-url", "http://127.0.0.1:8000/test-url2",
                "--model", "test-model"
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=True,
        )

        mock_airlock_backend.assert_has_calls(
            calls=[
                call(
                    base_url="http://127.0.0.1:8000/test-url1",
                    model="test-default-adapter",
                    api_key="EMPTY"
                ),
                call(
                    base_url="http://127.0.0.1:8000/test-url2",
                    model="test-model",
                    api_key="EMPTY"
                )

            ]
        )

    @patch('fifo_dev_dsl.common.llm_abstraction.OpenAICompatibleBackend')
    def test_create_openaicompatible_backend_with_custom_values_reasoning(self, mock_airlock_backend: MagicMock):
        """Test creating OpenAICompatibleBackend with custom values - dsl + reasoning only."""
        _ = parse_cli_and_create_backends(
            [
                "dsl=openai-compatible",
                "--adapter", "test-adapter",
                "--base-url", "http://127.0.0.1:8000/test-url1",
                "--api-key", "dummy-api-key1",

                "reasoning=openai-compatible",
                "--model", "test-model",
                "--base-url", "http://127.0.0.1:8000/test-url2",
                "--api-key", "dummy-api-key2",
            ],
            prog="test_program.py",
            description="Test description",
            default_adapter="test-default-adapter",
            require_reasoning=True,
        )

        mock_airlock_backend.assert_has_calls(
            calls=[
                call(
                    base_url="http://127.0.0.1:8000/test-url1",
                    model="test-adapter",
                    api_key="dummy-api-key1"
                ),
                call(
                    base_url="http://127.0.0.1:8000/test-url2",
                    model="test-model",
                    api_key="dummy-api-key2"
                )

            ]
        )

    def test_create_openai_backend_missing_base_url_raises_error_dsl(self):
        """Test that missing base_url for openai-compatible backend raises error."""

        with pytest.raises(SystemExit):
            _ = parse_cli_and_create_backends(
                [
                    "dsl=openai-compatible",
                    "--adapter", "test-adapter",
                    # "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key1",

                    "reasoning=openai-compatible",
                    "--model", "test-model",
                    "--base-url", "http://127.0.0.1:8000/test-url2",
                    "--api-key", "dummy-api-key2",
                ],
                prog="test_program.py",
                description="Test description",
                default_adapter="test-default-adapter",
                require_reasoning=True,
            )

    def test_create_openai_backend_missing_base_url_raises_error_reasoning(self):
        """Test that missing base_url for openai-compatible backend raises error."""

        with pytest.raises(SystemExit):
            _ = parse_cli_and_create_backends(
                [
                    "dsl=openai-compatible",
                    "--adapter", "test-adapter",
                    "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key1",

                    "reasoning=openai-compatible",
                    "--model", "test-model",
                    # "--base-url", "http://127.0.0.1:8000/test-url2",
                    "--api-key", "dummy-api-key2",
                ],
                prog="test_program.py",
                description="Test description",
                default_adapter="test-default-adapter",
                require_reasoning=True,
            )

    def test_create_backend_unknown_backend_type_raises_error_dsl(self):
        """Test that missing base_url for openai-compatible backend raises error."""

        with pytest.raises(SystemExit):
            _ = parse_cli_and_create_backends(
                [
                    "dsl=invalid-openai-compatible",
                    "--adapter", "test-adapter",
                    "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key1",

                    "reasoning=openai-compatible",
                    "--model", "test-model",
                    "--base-url", "http://127.0.0.1:8000/test-url2",
                    "--api-key", "dummy-api-key2",
                ],
                prog="test_program.py",
                description="Test description",
                default_adapter="test-default-adapter",
                require_reasoning=True,
            )

    def test_create_backend_unknown_backend_type_raises_error_reasoning(self):
        """Test that missing base_url for openai-compatible backend raises error."""

        with pytest.raises(SystemExit):
            _ = parse_cli_and_create_backends(
                [
                    "dsl=openai-compatible",
                    "--adapter", "test-adapter",
                    "--base-url", "http://127.0.0.1:8000/test-url1",
                    "--api-key", "dummy-api-key1",

                    "reasoning=reasoning-openai-compatible",
                    "--model", "test-model",
                    "--base-url", "http://127.0.0.1:8000/test-url2",
                    "--api-key", "dummy-api-key2",
                ],
                prog="test_program.py",
                description="Test description",
                default_adapter="test-default-adapter",
                require_reasoning=True,
            )

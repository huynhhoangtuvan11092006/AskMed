"""
tests/test_llm_client.py
Unit and live integration tests for OllamaLLMClient (src.generation.llm_client).
"""

from unittest.mock import MagicMock, patch
import pytest

from src.generation.llm_client import OllamaLLMClient


def test_llm_client_initialization_defaults():
    """Verify default configurations are correctly pulled from config.yaml."""
    client = OllamaLLMClient()
    assert client.model_name == "llama3.1:8b"
    assert client.temperature == 0.0
    assert client.top_p == 0.9
    assert client.max_tokens == 1024
    assert "11434" in client.base_url


def test_llm_client_build_options():
    """Verify inference options building with overrides."""
    client = OllamaLLMClient(temperature=0.0, top_p=0.9, max_tokens=1024)
    # Default options
    opts = client._build_options()
    assert opts == {"temperature": 0.0, "top_p": 0.9, "num_predict": 1024}

    # Overridden options
    opts_override = client._build_options(temperature=0.7, top_p=0.95, max_tokens=500)
    assert opts_override == {"temperature": 0.7, "top_p": 0.95, "num_predict": 500}


def test_llm_client_non_streaming_chat():
    """Verify non-streaming chat call formats messages and returns content string."""
    with patch("ollama.Client") as mock_ollama_client_class:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {
            "message": {"role": "assistant", "content": "Metformin is first-line."}
        }
        mock_ollama_client_class.return_value = mock_instance

        client = OllamaLLMClient()
        response = client.generate(
            prompt="What is the first-line medication for diabetes?",
            system_prompt="You are a medical bot.",
            stream=False,
        )

        assert response == "Metformin is first-line."
        mock_instance.chat.assert_called_once()
        call_kwargs = mock_instance.chat.call_args[1]
        assert call_kwargs["stream"] is False
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are a medical bot."},
            {"role": "user", "content": "What is the first-line medication for diabetes?"},
        ]
        assert call_kwargs["options"]["temperature"] == 0.0


def test_llm_client_streaming_chat():
    """Verify streaming chat yields tokens sequentially."""
    with patch("ollama.Client") as mock_ollama_client_class:
        mock_instance = MagicMock()
        # Mock streaming chunks
        mock_chunks = [
            {"message": {"role": "assistant", "content": "Metformin "}},
            {"message": {"role": "assistant", "content": "is "}},
            {"message": {"role": "assistant", "content": "first-line."}},
        ]
        mock_instance.chat.return_value = iter(mock_chunks)
        mock_ollama_client_class.return_value = mock_instance

        client = OllamaLLMClient()
        token_stream = client.generate(prompt="What is first-line?", stream=True)

        tokens = list(token_stream)
        assert tokens == ["Metformin ", "is ", "first-line."]
        assert "".join(tokens) == "Metformin is first-line."


def test_llm_client_health_check_handling():
    """Verify health check logic for both positive and error states."""
    with patch("ollama.Client") as mock_ollama_client_class:
        mock_instance = MagicMock()
        # Positive case: model exists
        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_list_resp = MagicMock()
        mock_list_resp.models = [mock_model]
        mock_instance.list.return_value = mock_list_resp
        mock_ollama_client_class.return_value = mock_instance

        client = OllamaLLMClient()
        assert client.is_healthy() is True

        # Negative case: connection exception
        mock_instance.list.side_effect = Exception("Connection refused")
        assert client.is_healthy() is False


def test_llm_client_live_inference():
    """Live test verifying deterministic inference with local Ollama daemon."""
    client = OllamaLLMClient()
    if not client.is_healthy():
        pytest.skip("Local Ollama daemon or llama3.1:8b not available for live test.")

    answer = client.generate(
        prompt="Reply with the single exact word: 'PULSE'",
        system_prompt="Be strictly minimal and follow instructions.",
        stream=False,
    )
    assert isinstance(answer, str)
    assert "PULSE" in answer.upper()

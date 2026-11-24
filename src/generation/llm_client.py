"""
src/generation/llm_client.py
Local Ollama LLM Client for Medical Text Generation in Ask_Med.
Wraps the Ollama Python SDK to communicate with local llama3.1:8b (or configured LLM).
Supports:
  - Deterministic clinical generation (temperature=0.0, top_p=0.9, num_predict=1024)
  - Both non-streaming (full response string) and streaming (token-by-token generator)
  - System prompt separation and chat-based messaging
  - Connection health verification and transparent error handling
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union

import ollama
from src.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """
    Client interface for interacting with local Ollama LLMs (e.g., llama3.1:8b).
    Designed specifically for evidence-based clinical reasoning where determinism
    and low hallucination rates are paramount.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ):
        """
        Initializes the Ollama LLM Client.

        Parameters:
            base_url: Ollama daemon URL (e.g. 'http://127.0.0.1:11434').
            model_name: LLM identifier (default from config: 'llama3.1:8b').
            temperature: Sampling temperature (default 0.0 for deterministic medical facts).
            top_p: Nucleus sampling probability (default 0.9).
            max_tokens: Maximum tokens to generate (num_predict in Ollama).
            timeout_seconds: HTTP request timeout.
        """
        cfg = get_config()
        self.base_url = base_url or cfg.models.ollama.base_url
        self.model_name = model_name or cfg.models.ollama.llm_model
        self.temperature = temperature if temperature is not None else cfg.models.ollama.temperature
        self.top_p = top_p if top_p is not None else cfg.models.ollama.top_p
        self.max_tokens = max_tokens or cfg.models.ollama.max_tokens
        self.timeout_seconds = timeout_seconds or cfg.models.ollama.timeout_seconds

        self.client = ollama.Client(host=self.base_url, timeout=self.timeout_seconds)
        logger.info(
            f"Initialized OllamaLLMClient with model='{self.model_name}', "
            f"host='{self.base_url}', temperature={self.temperature}, top_p={self.top_p}"
        )

    def is_healthy(self) -> bool:
        """
        Checks if the Ollama daemon is reachable and whether the configured model is installed.
        """
        try:
            response = self.client.list()
            models = [m.model for m in response.models]
            # Match model name with or without tag (e.g. 'llama3.1:8b' or 'llama3.1:latest')
            is_installed = any(
                self.model_name == m or self.model_name.split(":")[0] == m.split(":")[0]
                for m in models
            )
            if not is_installed:
                logger.warning(
                    f"Ollama server is active, but target model '{self.model_name}' was not found in installed models: {models}"
                )
            return is_installed
        except Exception as e:
            logger.error(f"Ollama health check failed on {self.base_url}: {e}")
            return False

    def _build_options(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Builds Ollama runtime inference options."""
        return {
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Sends a multi-turn or structured message list to the model.

        Parameters:
            messages: List of message dicts (e.g. [{'role': 'system', ...}, {'role': 'user', ...}]).
            stream: If True, returns an Iterator[str] yielding tokens. If False, returns complete string.
            temperature: Optional override for temperature.
            top_p: Optional override for top_p.
            max_tokens: Optional override for max_tokens.

        Returns:
            Complete text string if stream=False, or token generator if stream=True.
        """
        options = self._build_options(temperature, top_p, max_tokens)

        try:
            if not stream:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=False,
                )
                return response["message"]["content"]
            else:
                raw_stream = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=True,
                )

                def generator() -> Iterator[str]:
                    for chunk in raw_stream:
                        msg = chunk.get("message", {})
                        token = msg.get("content", "")
                        if token:
                            yield token

                return generator()

        except Exception as e:
            logger.error(f"Error calling Ollama LLM chat ({self.model_name}): {e}")
            raise RuntimeError(f"Ollama LLM generation error: {e}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Convenience method to generate an answer from a user prompt with an optional system prompt.

        Parameters:
            prompt: User question or clinical query containing retrieved context.
            system_prompt: Optional instructions governing model persona and citation rules.
            stream: Whether to stream tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response string or token stream iterator.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Testing OllamaLLMClient (Step 4.1 Demo)")
    print("=" * 60)

    llm = OllamaLLMClient()

    print("\n[1/3] Checking Ollama service health...")
    healthy = llm.is_healthy()
    print(f"-> Ollama Health Status: {'ONLINE & READY' if healthy else 'OFFLINE / MODEL MISSING'}")
    if not healthy:
        print("Please ensure Ollama is running and 'llama3.1:8b' is pulled.")
        sys.exit(1)

    system_instruction = (
        "You are an evidence-based clinical AI assistant. Answer concisely in 2 sentences."
    )
    user_prompt = "What is the recommended first-line pharmacological treatment for type 2 diabetes?"

    print("\n[2/3] Testing Non-Streaming Generation (temperature=0.0)...")
    answer = llm.generate(prompt=user_prompt, system_prompt=system_instruction, stream=False)
    print(f"-> Non-Streaming Response:\n{answer}\n")

    print("[3/3] Testing Streaming Generation (token-by-token)...")
    print("-> Streaming Response: ", end="", flush=True)
    stream_iter = llm.generate(prompt=user_prompt, system_prompt=system_instruction, stream=True)
    for token in stream_iter:
        print(token, end="", flush=True)
    print("\n\nTest completed successfully!")

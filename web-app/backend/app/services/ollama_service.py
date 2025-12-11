"""
Ollama service for LLM communication with streaming support.
Provides integration with local Ollama instance for response generation.
"""
import asyncio
import logging
import os
from typing import AsyncGenerator, List, Optional, Dict, Any
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

# Default Ollama configuration
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 120.0  # seconds


@dataclass
class OllamaModel:
    """Represents an Ollama model."""
    name: str
    size: Optional[int] = None
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # 'user' | 'assistant' | 'system'
    content: str


class OllamaServiceError(Exception):
    """Base exception for Ollama service errors."""
    pass


class OllamaConnectionError(OllamaServiceError):
    """Raised when Ollama is not available."""
    pass


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when the requested model is not found."""
    pass


class OllamaService:
    """
    Service for communicating with Ollama local LLM.
    
    Provides streaming response generation, model listing, and health checks.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT
    ):
        """
        Initialize Ollama service.
        
        Args:
            host: Ollama server URL (defaults to OLLAMA_HOST env var or localhost:11434)
            timeout: Request timeout in seconds
        """
        self.host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to Ollama at {self.host}")
            return False
        except httpx.TimeoutException:
            logger.warning(f"Timeout connecting to Ollama at {self.host}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama health: {e}")
            return False
    
    async def list_models(self) -> List[OllamaModel]:
        """
        Fetch available models from Ollama.
        
        Returns:
            List of available OllamaModel objects
            
        Raises:
            OllamaConnectionError: If Ollama is not available
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            
            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Failed to list models: HTTP {response.status_code}"
                )
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                model = OllamaModel(
                    name=model_data.get("name", ""),
                    size=model_data.get("size"),
                    digest=model_data.get("digest"),
                    modified_at=model_data.get("modified_at"),
                    details=model_data.get("details")
                )
                models.append(model)
            
            logger.info(f"Found {len(models)} Ollama models")
            return models
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise OllamaConnectionError(
                f"Ollama is not running. Please start Ollama at {self.host}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to Ollama: {e}")
            raise OllamaConnectionError(
                f"Timeout connecting to Ollama at {self.host}"
            ) from e
    
    async def generate_stream(
        self,
        question: str,
        model: str = DEFAULT_MODEL,
        conversation_history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama for a given question.
        
        Args:
            question: The user's question
            model: Ollama model name to use
            conversation_history: Previous messages for context
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter
            
        Yields:
            Individual tokens as they are generated
            
        Raises:
            OllamaConnectionError: If Ollama is not available
            OllamaModelNotFoundError: If the model is not found
        """
        # Build messages array for chat endpoint
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        try:
            client = await self._get_client()
            
            async with client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{model}' not found. Please pull it first with: ollama pull {model}"
                    )
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise OllamaServiceError(
                        f"Ollama error: HTTP {response.status_code} - {error_text.decode()}"
                    )
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        import json
                        data = json.loads(line)
                        
                        # Extract token from response
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            if token:
                                yield token
                        
                        # Check if generation is complete
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama response: {line}")
                        continue
                        
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise OllamaConnectionError(
                f"Ollama is not running. Please start Ollama at {self.host}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Timeout during generation: {e}")
            raise OllamaConnectionError(
                f"Response generation timed out after {self.timeout}s"
            ) from e
        except OllamaServiceError:
            raise
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise OllamaServiceError(f"Generation failed: {str(e)}") from e
    
    async def generate(
        self,
        question: str,
        model: str = DEFAULT_MODEL,
        conversation_history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a complete response from Ollama (non-streaming).
        
        Args:
            question: The user's question
            model: Ollama model name to use
            conversation_history: Previous messages for context
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter
            
        Returns:
            Complete generated response
            
        Raises:
            OllamaConnectionError: If Ollama is not available
            OllamaModelNotFoundError: If the model is not found
        """
        tokens = []
        async for token in self.generate_stream(
            question=question,
            model=model,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p
        ):
            tokens.append(token)
        
        return "".join(tokens)


# Singleton instance for dependency injection
_ollama_service: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:
    """Get or create the Ollama service singleton."""
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service

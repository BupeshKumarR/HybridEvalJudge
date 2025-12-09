"""
Base interface for API-based judge clients.

This module defines the abstract base class that all API judge clients
(Groq, Gemini, etc.) must implement to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_judge_auditor.models import Issue


@dataclass
class JudgeVerdict:
    """
    Verdict from an API-based judge evaluation.
    
    This is similar to JudgeResult but specifically for API judges,
    with additional metadata about the API call.
    
    Attributes:
        judge_name: Name of the judge (e.g., "groq-llama-3.1-70b")
        score: Factual accuracy score (0-100)
        confidence: Judge's confidence in the evaluation (0.0-1.0)
        reasoning: Chain-of-thought explanation
        issues: List of detected issues
        metadata: Additional metadata (response time, tokens used, etc.)
    """
    
    judge_name: str
    score: float
    confidence: float
    reasoning: str
    issues: List[Issue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseJudgeClient(ABC):
    """
    Abstract base class for API-based judge clients.
    
    All judge clients (Groq, Gemini, etc.) must implement this interface
    to ensure consistent behavior across different APIs.
    """
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize the judge client.
        
        Args:
            api_key: API key for authentication
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def evaluate(
        self,
        source_text: str,
        candidate_output: str,
        task: str = "factual_accuracy"
    ) -> JudgeVerdict:
        """
        Evaluate candidate output against source text.
        
        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            task: Evaluation task type (e.g., "factual_accuracy", "bias_detection")
        
        Returns:
            JudgeVerdict with score, reasoning, and detected issues
        
        Raises:
            Exception: If API call fails
        """
        pass
    
    @abstractmethod
    def _format_prompt(
        self,
        source_text: str,
        candidate_output: str,
        task: str
    ) -> str:
        """
        Format the evaluation prompt for this specific API.
        
        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            task: Evaluation task type
        
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: Any) -> JudgeVerdict:
        """
        Parse the API response into a structured JudgeVerdict.
        
        Args:
            response: Raw response from the API
        
        Returns:
            Parsed JudgeVerdict
        
        Raises:
            ValueError: If response cannot be parsed
        """
        pass
    
    def get_judge_name(self) -> str:
        """
        Get the name of this judge for identification.
        
        Returns:
            Judge name string
        """
        return f"{self.__class__.__name__.replace('JudgeClient', '').lower()}-{self.model}"

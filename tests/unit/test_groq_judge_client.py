"""
Unit tests for the GroqJudgeClient component.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys

from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqAPIError,
    GroqRateLimitError,
    GroqAuthenticationError,
    GroqNetworkError,
)
from llm_judge_auditor.components.base_judge_client import JudgeVerdict
from llm_judge_auditor.models import Issue, IssueSeverity, IssueType


class TestGroqJudgeClient:
    """Test suite for GroqJudgeClient."""
    
    @patch('groq.Groq')
    def test_initialization(self, mock_groq):
        """Test that GroqJudgeClient initializes correctly."""
        client = GroqJudgeClient(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.model == "llama-3.3-70b-versatile"
        assert client.max_retries == 2
        assert client.base_delay == 1.0
        assert client.timeout == 30
        mock_groq.assert_called_once_with(api_key="test-key", timeout=30)
    
    @patch('groq.Groq')
    def test_initialization_custom_params(self, mock_groq):
        """Test initialization with custom parameters."""
        client = GroqJudgeClient(
            api_key="test-key",
            model="custom-model",
            max_retries=5,
            base_delay=2.0,
            timeout=60
        )
        
        assert client.model == "custom-model"
        assert client.max_retries == 5
        assert client.base_delay == 2.0
        assert client.timeout == 60
    
    def test_initialization_missing_groq_package(self):
        """Test that initialization fails gracefully when groq package is missing."""
        # Temporarily remove groq from sys.modules to simulate missing package
        groq_module = sys.modules.get('groq')
        if 'groq' in sys.modules:
            del sys.modules['groq']
        
        try:
            with patch.dict('sys.modules', {'groq': None}):
                with pytest.raises(ImportError, match="groq package not installed"):
                    GroqJudgeClient(api_key="test-key")
        finally:
            # Restore groq module
            if groq_module is not None:
                sys.modules['groq'] = groq_module
    
    @patch('groq.Groq')
    def test_format_prompt_factual_accuracy(self, mock_groq):
        """Test prompt formatting for factual accuracy task."""
        client = GroqJudgeClient(api_key="test-key")
        
        prompt = client._format_prompt(
            source_text="Paris is the capital of France.",
            candidate_output="Paris is the capital of Germany.",
            task="factual_accuracy"
        )
        
        assert "Paris is the capital of France." in prompt
        assert "Paris is the capital of Germany." in prompt
        assert "factual accuracy" in prompt.lower()
        assert "JSON" in prompt
        assert "score" in prompt.lower()
    
    @patch('groq.Groq')
    def test_format_prompt_bias_detection(self, mock_groq):
        """Test prompt formatting for bias detection task."""
        client = GroqJudgeClient(api_key="test-key")
        
        prompt = client._format_prompt(
            source_text="",
            candidate_output="Some text to check for bias.",
            task="bias_detection"
        )
        
        assert "Some text to check for bias." in prompt
        assert "bias" in prompt.lower()
        assert "JSON" in prompt
    
    @patch('groq.Groq')
    def test_parse_response_success(self, mock_groq):
        """Test successful response parsing."""
        client = GroqJudgeClient(api_key="test-key")
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 85.5,
            "confidence": 0.9,
            "reasoning": "The output is mostly accurate.",
            "issues": [
                {
                    "severity": "low",
                    "type": "factual_error",
                    "description": "Minor inaccuracy",
                    "location": "line 1"
                }
            ]
        })
        mock_response.usage.total_tokens = 150
        
        verdict = client._parse_response(mock_response)
        
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.score == 85.5
        assert verdict.confidence == 0.9
        assert verdict.reasoning == "The output is mostly accurate."
        assert len(verdict.issues) == 1
        assert verdict.issues[0].severity == IssueSeverity.LOW
        assert verdict.issues[0].type == IssueType.FACTUAL_ERROR
        assert verdict.metadata["tokens_used"] == 150
    
    @patch('groq.Groq')
    def test_parse_response_clamps_score(self, mock_groq):
        """Test that scores are clamped to 0-100 range."""
        client = GroqJudgeClient(api_key="test-key")
        
        # Test score > 100
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 150,
            "confidence": 0.9,
            "reasoning": "Test"
        })
        mock_response.usage.total_tokens = 100
        
        verdict = client._parse_response(mock_response)
        assert verdict.score == 100.0
        
        # Test score < 0
        mock_response.choices[0].message.content = json.dumps({
            "score": -50,
            "confidence": 0.9,
            "reasoning": "Test"
        })
        
        verdict = client._parse_response(mock_response)
        assert verdict.score == 0.0
    
    @patch('groq.Groq')
    def test_parse_response_malformed_json(self, mock_groq):
        """Test fallback parsing for malformed JSON."""
        client = GroqJudgeClient(api_key="test-key")
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not valid JSON but score: 75"
        
        verdict = client._parse_response(mock_response)
        
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.score == 75.0
        assert verdict.confidence == 0.5  # Low confidence for fallback
        assert verdict.metadata.get("fallback_parsing") is True
    
    @patch('groq.Groq')
    def test_fallback_parse_extracts_score(self, mock_groq):
        """Test that fallback parser can extract scores from various formats."""
        client = GroqJudgeClient(api_key="test-key")
        
        # Test various score formats
        test_cases = [
            ('"score": 85', 85.0),
            ('score: 92.5', 92.5),
            ('Score: 70', 70.0),
            ('45 / 100', 45.0),
        ]
        
        for text, expected_score in test_cases:
            verdict = client._fallback_parse(text)
            assert verdict.score == expected_score
    
    @patch('groq.Groq')
    def test_evaluate_success(self, mock_groq):
        """Test successful evaluation."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 80,
            "confidence": 0.85,
            "reasoning": "Good accuracy",
            "issues": []
        })
        mock_response.usage.total_tokens = 200
        
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        client = GroqJudgeClient(api_key="test-key")
        verdict = client.evaluate(
            source_text="Test source",
            candidate_output="Test output",
            task="factual_accuracy"
        )
        
        assert verdict.score == 80
        assert verdict.confidence == 0.85
        assert "response_time_seconds" in verdict.metadata
        assert verdict.metadata["model"] == "llama-3.3-70b-versatile"
        assert verdict.metadata["task"] == "factual_accuracy"
    
    @patch('groq.Groq')
    def test_call_api_with_retry_authentication_error(self, mock_groq):
        """Test that authentication errors are not retried."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Simulate authentication error
        mock_client_instance.chat.completions.create.side_effect = Exception("401 unauthorized")
        
        client = GroqJudgeClient(api_key="test-key")
        
        with pytest.raises(GroqAuthenticationError, match="authentication failed"):
            client._call_api_with_retry("test prompt")
        
        # Should only be called once (no retries)
        assert mock_client_instance.chat.completions.create.call_count == 1
    
    @patch('groq.Groq')
    @patch('time.sleep')
    def test_call_api_with_retry_rate_limit(self, mock_sleep, mock_groq):
        """Test retry logic for rate limit errors."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # First two calls fail with rate limit, third succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 80}'
        
        mock_client_instance.chat.completions.create.side_effect = [
            Exception("429 rate limit exceeded"),
            Exception("429 rate limit exceeded"),
            mock_response
        ]
        
        client = GroqJudgeClient(api_key="test-key", max_retries=2)
        response = client._call_api_with_retry("test prompt")
        
        assert response == mock_response
        assert mock_client_instance.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries
    
    @patch('groq.Groq')
    @patch('time.sleep')
    def test_call_api_with_retry_rate_limit_exhausted(self, mock_sleep, mock_groq):
        """Test that rate limit errors raise exception after max retries."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # All calls fail with rate limit
        mock_client_instance.chat.completions.create.side_effect = Exception("429 rate limit exceeded")
        
        client = GroqJudgeClient(api_key="test-key", max_retries=2)
        
        with pytest.raises(GroqRateLimitError, match="rate limit exceeded"):
            client._call_api_with_retry("test prompt")
        
        assert mock_client_instance.chat.completions.create.call_count == 3  # Initial + 2 retries
    
    @patch('groq.Groq')
    @patch('time.sleep')
    def test_call_api_with_retry_network_error(self, mock_sleep, mock_groq):
        """Test retry logic for network errors."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # First call fails with network error, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 80}'
        
        mock_client_instance.chat.completions.create.side_effect = [
            Exception("connection timeout"),
            mock_response
        ]
        
        client = GroqJudgeClient(api_key="test-key")
        response = client._call_api_with_retry("test prompt")
        
        assert response == mock_response
        assert mock_client_instance.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1
    
    @patch('groq.Groq')
    @patch('time.sleep')
    def test_call_api_with_retry_exponential_backoff(self, mock_sleep, mock_groq):
        """Test that exponential backoff is applied correctly."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # All calls fail to test backoff delays
        mock_client_instance.chat.completions.create.side_effect = Exception("500 server error")
        
        client = GroqJudgeClient(api_key="test-key", max_retries=2, base_delay=1.0)
        
        with pytest.raises(GroqNetworkError):
            client._call_api_with_retry("test prompt")
        
        # Check that sleep was called with exponentially increasing delays
        assert mock_sleep.call_count == 2
        # First retry: 1.0 * 2^0 = 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)
    
    @patch('groq.Groq')
    def test_call_api_with_retry_client_error_no_retry(self, mock_groq):
        """Test that 4xx client errors (except 429) are not retried."""
        mock_client_instance = Mock()
        mock_groq.return_value = mock_client_instance
        
        # Simulate 400 error
        mock_client_instance.chat.completions.create.side_effect = Exception("400 bad request")
        
        client = GroqJudgeClient(api_key="test-key")
        
        with pytest.raises(GroqAPIError, match="client error"):
            client._call_api_with_retry("test prompt")
        
        # Should only be called once (no retries)
        assert mock_client_instance.chat.completions.create.call_count == 1
    
    @patch('groq.Groq')
    def test_extract_retry_after(self, mock_groq):
        """Test extraction of retry-after value from exceptions."""
        client = GroqJudgeClient(api_key="test-key")
        
        # Test with retry_after attribute
        exception = Exception("Rate limit")
        exception.retry_after = 5.0
        assert client._extract_retry_after(exception) == 5.0
        
        # Test with retry-after in message
        exception = Exception("Rate limit exceeded. Retry after 10 seconds")
        assert client._extract_retry_after(exception) == 10.0
        
        # Test with no retry-after info
        exception = Exception("Some error")
        assert client._extract_retry_after(exception) is None
    
    @patch('groq.Groq')
    def test_get_judge_name(self, mock_groq):
        """Test that judge name is formatted correctly."""
        client = GroqJudgeClient(api_key="test-key", model="llama-3.3-70b-versatile")
        assert client.get_judge_name() == "groq-llama-3.3-70b-versatile"
        
        client = GroqJudgeClient(api_key="test-key", model="custom-model")
        assert client.get_judge_name() == "groq-custom-model"
    
    @patch('groq.Groq')
    def test_parse_response_handles_missing_fields(self, mock_groq):
        """Test that parser handles responses with missing optional fields."""
        client = GroqJudgeClient(api_key="test-key")
        
        # Minimal response with only score
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 75
        })
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        
        verdict = client._parse_response(mock_response)
        
        assert verdict.score == 75.0
        assert verdict.confidence == 0.7  # Default
        assert verdict.reasoning == "No reasoning provided"  # Default
        assert len(verdict.issues) == 0
    
    @patch('groq.Groq')
    def test_parse_response_handles_invalid_issue(self, mock_groq):
        """Test that parser skips invalid issues gracefully."""
        client = GroqJudgeClient(api_key="test-key")
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": 80,
            "confidence": 0.9,
            "reasoning": "Test",
            "issues": [
                {
                    "severity": "low",
                    "type": "factual_error",
                    "description": "Valid issue"
                },
                {
                    # Invalid issue - missing required fields
                    "invalid": "data"
                },
                {
                    "severity": "high",
                    "type": "hallucination",
                    "description": "Another valid issue"
                }
            ]
        })
        mock_response.usage.total_tokens = 100
        
        verdict = client._parse_response(mock_response)
        
        # Should have 3 issues - 2 valid ones and 1 with default values from invalid data
        assert len(verdict.issues) == 3
        assert verdict.issues[0].description == "Valid issue"
        assert verdict.issues[1].description == "Issue detected"  # Default for invalid issue
        assert verdict.issues[2].description == "Another valid issue"

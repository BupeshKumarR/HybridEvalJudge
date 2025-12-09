"""
Unit tests for APIJudgeEnsemble component.

Tests the ensemble coordination, parallel execution, and aggregation logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.components.base_judge_client import JudgeVerdict
from llm_judge_auditor.config import ToolkitConfig
from llm_judge_auditor.models import Issue, IssueSeverity, IssueType


@pytest.fixture
def mock_api_key_manager():
    """Create a mock API key manager with both keys available."""
    manager = Mock(spec=APIKeyManager)
    manager.groq_key = "test-groq-key"
    manager.gemini_key = "test-gemini-key"
    return manager


@pytest.fixture
def mock_api_key_manager_groq_only():
    """Create a mock API key manager with only Groq key."""
    manager = Mock(spec=APIKeyManager)
    manager.groq_key = "test-groq-key"
    manager.gemini_key = None
    return manager


@pytest.fixture
def mock_api_key_manager_no_keys():
    """Create a mock API key manager with no keys."""
    manager = Mock(spec=APIKeyManager)
    manager.groq_key = None
    manager.gemini_key = None
    return manager


@pytest.fixture
def mock_config():
    """Create a mock toolkit config."""
    return Mock(spec=ToolkitConfig)


@pytest.fixture
def sample_verdict_groq():
    """Create a sample Groq verdict."""
    return JudgeVerdict(
        judge_name="groq-llama-3.3-70b-versatile",
        score=85.0,
        confidence=0.9,
        reasoning="The output is mostly accurate with minor issues.",
        issues=[
            Issue(
                type=IssueType.FACTUAL_ERROR,
                severity=IssueSeverity.LOW,
                description="Minor factual discrepancy",
                evidence=["sentence 3"]
            )
        ],
        metadata={"response_time_seconds": 2.5}
    )


@pytest.fixture
def sample_verdict_gemini():
    """Create a sample Gemini verdict."""
    return JudgeVerdict(
        judge_name="gemini-gemini-1.5-flash",
        score=80.0,
        confidence=0.85,
        reasoning="Good accuracy overall with some unsupported claims.",
        issues=[
            Issue(
                type=IssueType.UNSUPPORTED_CLAIM,
                severity=IssueSeverity.MEDIUM,
                description="Claim not supported by source",
                evidence=["paragraph 2"]
            )
        ],
        metadata={"response_time_seconds": 3.0}
    )


class TestAPIJudgeEnsembleInitialization:
    """Test ensemble initialization with different API key configurations."""
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_initialize_with_both_keys(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager
    ):
        """Test initialization with both API keys available."""
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager)
        
        assert ensemble.get_judge_count() == 2
        assert mock_groq_class.called
        assert mock_gemini_class.called
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_initialize_with_groq_only(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager_groq_only
    ):
        """Test initialization with only Groq key."""
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager_groq_only)
        
        assert ensemble.get_judge_count() == 1
        assert mock_groq_class.called
        assert not mock_gemini_class.called
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_initialize_with_no_keys(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager_no_keys
    ):
        """Test initialization with no API keys."""
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager_no_keys)
        
        assert ensemble.get_judge_count() == 0
        assert not mock_groq_class.called
        assert not mock_gemini_class.called


class TestAPIJudgeEnsembleEvaluation:
    """Test ensemble evaluation functionality."""
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_evaluate_parallel_success(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager,
        sample_verdict_groq,
        sample_verdict_gemini
    ):
        """Test successful parallel evaluation with both judges."""
        # Setup mock judges
        mock_groq = MagicMock()
        mock_groq.evaluate.return_value = sample_verdict_groq
        mock_groq.get_judge_name.return_value = "groq-llama-3.3-70b-versatile"
        mock_groq_class.return_value = mock_groq
        
        mock_gemini = MagicMock()
        mock_gemini.evaluate.return_value = sample_verdict_gemini
        mock_gemini.get_judge_name.return_value = "gemini-gemini-1.5-flash"
        mock_gemini_class.return_value = mock_gemini
        
        # Create ensemble and evaluate
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager, parallel_execution=True)
        verdicts = ensemble.evaluate(
            source_text="The sky is blue.",
            candidate_output="The sky is blue and beautiful.",
            task="factual_accuracy"
        )
        
        assert len(verdicts) == 2
        assert any(v.judge_name == "groq-llama-3.3-70b-versatile" for v in verdicts)
        assert any(v.judge_name == "gemini-gemini-1.5-flash" for v in verdicts)
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_evaluate_sequential_success(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager,
        sample_verdict_groq,
        sample_verdict_gemini
    ):
        """Test successful sequential evaluation with both judges."""
        # Setup mock judges
        mock_groq = MagicMock()
        mock_groq.evaluate.return_value = sample_verdict_groq
        mock_groq.get_judge_name.return_value = "groq-llama-3.3-70b-versatile"
        mock_groq_class.return_value = mock_groq
        
        mock_gemini = MagicMock()
        mock_gemini.evaluate.return_value = sample_verdict_gemini
        mock_gemini.get_judge_name.return_value = "gemini-gemini-1.5-flash"
        mock_gemini_class.return_value = mock_gemini
        
        # Create ensemble and evaluate
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager, parallel_execution=False)
        verdicts = ensemble.evaluate(
            source_text="The sky is blue.",
            candidate_output="The sky is blue and beautiful.",
            task="factual_accuracy"
        )
        
        assert len(verdicts) == 2
        assert mock_groq.evaluate.called
        assert mock_gemini.evaluate.called
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_evaluate_partial_failure(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager,
        sample_verdict_groq
    ):
        """Test evaluation when one judge fails but another succeeds."""
        # Setup mock judges - Groq succeeds, Gemini fails
        mock_groq = MagicMock()
        mock_groq.evaluate.return_value = sample_verdict_groq
        mock_groq.get_judge_name.return_value = "groq-llama-3.3-70b-versatile"
        mock_groq_class.return_value = mock_groq
        
        mock_gemini = MagicMock()
        mock_gemini.evaluate.side_effect = Exception("API error")
        mock_gemini.get_judge_name.return_value = "gemini-gemini-1.5-flash"
        mock_gemini_class.return_value = mock_gemini
        
        # Create ensemble and evaluate
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager)
        verdicts = ensemble.evaluate(
            source_text="The sky is blue.",
            candidate_output="The sky is blue and beautiful.",
            task="factual_accuracy"
        )
        
        # Should still get one verdict from successful judge
        assert len(verdicts) == 1
        assert verdicts[0].judge_name == "groq-llama-3.3-70b-versatile"
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_evaluate_all_judges_fail(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager
    ):
        """Test evaluation when all judges fail."""
        # Setup mock judges - both fail
        mock_groq = MagicMock()
        mock_groq.evaluate.side_effect = Exception("Groq API error")
        mock_groq.get_judge_name.return_value = "groq-llama-3.3-70b-versatile"
        mock_groq_class.return_value = mock_groq
        
        mock_gemini = MagicMock()
        mock_gemini.evaluate.side_effect = Exception("Gemini API error")
        mock_gemini.get_judge_name.return_value = "gemini-gemini-1.5-flash"
        mock_gemini_class.return_value = mock_gemini
        
        # Create ensemble and evaluate
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager)
        
        with pytest.raises(RuntimeError, match="All judges failed"):
            ensemble.evaluate(
                source_text="The sky is blue.",
                candidate_output="The sky is blue and beautiful.",
                task="factual_accuracy"
            )
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_evaluate_no_judges_available(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager_no_keys
    ):
        """Test evaluation when no judges are available."""
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager_no_keys)
        
        with pytest.raises(RuntimeError, match="No judges available"):
            ensemble.evaluate(
                source_text="The sky is blue.",
                candidate_output="The sky is blue and beautiful.",
                task="factual_accuracy"
            )


class TestAPIJudgeEnsembleAggregation:
    """Test ensemble aggregation functionality."""
    
    def test_aggregate_verdicts_two_judges(self, sample_verdict_groq, sample_verdict_gemini):
        """Test aggregating verdicts from two judges."""
        verdicts = [sample_verdict_groq, sample_verdict_gemini]
        
        # Create a minimal ensemble just for aggregation
        manager = Mock(spec=APIKeyManager)
        manager.groq_key = None
        manager.gemini_key = None
        config = Mock(spec=ToolkitConfig)
        ensemble = APIJudgeEnsemble(config, manager)
        
        consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts)
        
        # Consensus should be mean of 85.0 and 80.0
        assert consensus == 82.5
        assert individual["groq-llama-3.3-70b-versatile"] == 85.0
        assert individual["gemini-gemini-1.5-flash"] == 80.0
        assert disagreement > 0  # Should have some variance
    
    def test_aggregate_verdicts_single_judge(self, sample_verdict_groq):
        """Test aggregating verdicts from a single judge."""
        verdicts = [sample_verdict_groq]
        
        manager = Mock(spec=APIKeyManager)
        manager.groq_key = None
        manager.gemini_key = None
        config = Mock(spec=ToolkitConfig)
        ensemble = APIJudgeEnsemble(config, manager)
        
        consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts)
        
        assert consensus == 85.0
        assert individual["groq-llama-3.3-70b-versatile"] == 85.0
        assert disagreement == 0.0  # No variance with single judge
    
    def test_aggregate_verdicts_empty(self):
        """Test aggregating empty verdicts list."""
        manager = Mock(spec=APIKeyManager)
        manager.groq_key = None
        manager.gemini_key = None
        config = Mock(spec=ToolkitConfig)
        ensemble = APIJudgeEnsemble(config, manager)
        
        consensus, individual, disagreement = ensemble.aggregate_verdicts([])
        
        assert consensus == 0.0
        assert individual == {}
        assert disagreement == 0.0


class TestAPIJudgeEnsembleDisagreement:
    """Test disagreement detection functionality."""
    
    def test_identify_disagreements_low_variance(self, sample_verdict_groq):
        """Test disagreement detection with low variance."""
        # Create two verdicts with similar scores
        verdict1 = sample_verdict_groq
        verdict2 = JudgeVerdict(
            judge_name="gemini-gemini-1.5-flash",
            score=83.0,  # Close to 85.0
            confidence=0.9,
            reasoning="Similar assessment",
            issues=[],
            metadata={}
        )
        
        manager = Mock(spec=APIKeyManager)
        manager.groq_key = None
        manager.gemini_key = None
        config = Mock(spec=ToolkitConfig)
        ensemble = APIJudgeEnsemble(config, manager)
        
        disagreement = ensemble.identify_disagreements([verdict1, verdict2], threshold=20.0)
        
        assert not disagreement["has_disagreement"]
        assert disagreement["variance"] < 20.0
        assert len(disagreement["outliers"]) == 0
    
    def test_identify_disagreements_high_variance(self, sample_verdict_groq):
        """Test disagreement detection with high variance."""
        # Create two verdicts with very different scores
        verdict1 = sample_verdict_groq  # score=85.0
        verdict2 = JudgeVerdict(
            judge_name="gemini-gemini-1.5-flash",
            score=30.0,  # Very different from 85.0
            confidence=0.9,
            reasoning="Poor quality",
            issues=[],
            metadata={}
        )
        
        manager = Mock(spec=APIKeyManager)
        manager.groq_key = None
        manager.gemini_key = None
        config = Mock(spec=ToolkitConfig)
        ensemble = APIJudgeEnsemble(config, manager)
        
        disagreement = ensemble.identify_disagreements([verdict1, verdict2], threshold=20.0)
        
        assert disagreement["has_disagreement"]
        assert disagreement["variance"] > 20.0
        assert disagreement["score_range"] == (30.0, 85.0)
    
    def test_identify_disagreements_with_outliers(self):
        """Test disagreement detection with outlier identification."""
        # Need at least 4 judges for outlier detection to work properly
        verdicts = [
            JudgeVerdict("judge1", 80.0, 0.9, "Good", [], {}),
            JudgeVerdict("judge2", 82.0, 0.9, "Good", [], {}),
            JudgeVerdict("judge3", 81.0, 0.9, "Good", [], {}),
            JudgeVerdict("judge4", 30.0, 0.9, "Poor", [], {}),  # Outlier
        ]
        
        manager = Mock(spec=APIKeyManager)
        manager.groq_key = None
        manager.gemini_key = None
        config = Mock(spec=ToolkitConfig)
        ensemble = APIJudgeEnsemble(config, manager)
        
        disagreement = ensemble.identify_disagreements(verdicts, threshold=20.0)
        
        assert disagreement["has_disagreement"]
        assert "judge4" in disagreement["outliers"]


class TestAPIJudgeEnsembleUtilities:
    """Test utility methods."""
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_get_judge_count(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager
    ):
        """Test getting judge count."""
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager)
        assert ensemble.get_judge_count() == 2
    
    @patch('llm_judge_auditor.components.api_judge_ensemble.GroqJudgeClient')
    @patch('llm_judge_auditor.components.api_judge_ensemble.GeminiJudgeClient')
    def test_get_judge_names(
        self,
        mock_gemini_class,
        mock_groq_class,
        mock_config,
        mock_api_key_manager
    ):
        """Test getting judge names."""
        mock_groq = MagicMock()
        mock_groq.get_judge_name.return_value = "groq-llama-3.3-70b-versatile"
        mock_groq_class.return_value = mock_groq
        
        mock_gemini = MagicMock()
        mock_gemini.get_judge_name.return_value = "gemini-gemini-1.5-flash"
        mock_gemini_class.return_value = mock_gemini
        
        ensemble = APIJudgeEnsemble(mock_config, mock_api_key_manager)
        names = ensemble.get_judge_names()
        
        assert len(names) == 2
        assert "groq-llama-3.3-70b-versatile" in names
        assert "gemini-gemini-1.5-flash" in names

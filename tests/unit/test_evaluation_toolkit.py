"""
Unit tests for the EvaluationToolkit orchestrator.

Tests the main entry point and multi-stage pipeline integration.
"""

import pytest
from unittest.mock import MagicMock, Mock, patch

from llm_judge_auditor.config import ToolkitConfig, AggregationStrategy
from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
from llm_judge_auditor.models import (
    EvaluationResult,
    Issue,
    IssueType,
    IssueSeverity,
    JudgeResult,
    Passage,
    Verdict,
    VerdictLabel,
)


@pytest.fixture
def mock_config():
    """Create a minimal test configuration."""
    config = ToolkitConfig(
        verifier_model="test-verifier",
        judge_models=["test-judge-1", "test-judge-2"],
        quantize=False,
        enable_retrieval=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        batch_size=1,
        max_length=512,
    )
    return config


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager."""
    manager = MagicMock()
    manager.verify_models_ready.return_value = True
    
    # Mock verifier
    mock_verifier_model = MagicMock()
    mock_verifier_tokenizer = MagicMock()
    manager.get_verifier.return_value = (mock_verifier_model, mock_verifier_tokenizer)
    
    # Mock judges
    mock_judge_model = MagicMock()
    mock_judge_tokenizer = MagicMock()
    manager.get_all_judges.return_value = {
        "test-judge-1": (mock_judge_model, mock_judge_tokenizer),
        "test-judge-2": (mock_judge_model, mock_judge_tokenizer),
    }
    
    return manager


@pytest.fixture
def mock_components():
    """Create mocks for all toolkit components."""
    with patch("llm_judge_auditor.evaluation_toolkit.ModelManager") as mock_mm, \
         patch("llm_judge_auditor.evaluation_toolkit.RetrievalComponent") as mock_rc, \
         patch("llm_judge_auditor.evaluation_toolkit.SpecializedVerifier") as mock_sv, \
         patch("llm_judge_auditor.evaluation_toolkit.PromptManager") as mock_pm, \
         patch("llm_judge_auditor.evaluation_toolkit.JudgeEnsemble") as mock_je, \
         patch("llm_judge_auditor.evaluation_toolkit.AggregationEngine") as mock_ae, \
         patch("llm_judge_auditor.components.api_key_manager.APIKeyManager") as mock_akm, \
         patch("llm_judge_auditor.components.api_judge_ensemble.APIJudgeEnsemble") as mock_aje:
        
        # Configure ModelManager mock
        mock_mm_instance = MagicMock()
        mock_mm_instance.verify_models_ready.return_value = True
        mock_mm_instance.get_verifier.return_value = (MagicMock(), MagicMock())
        mock_mm_instance.get_all_judges.return_value = {
            "test-judge-1": (MagicMock(), MagicMock()),
        }
        mock_mm_instance.get_model_info.return_value = {"verifier": "test-verifier"}
        # Make load_verifier and load_judge succeed silently
        mock_mm_instance.load_verifier.return_value = None
        mock_mm_instance.load_judge.return_value = None
        mock_mm.return_value = mock_mm_instance
        
        # Configure RetrievalComponent mock
        mock_rc_instance = MagicMock()
        mock_rc_instance.fallback_mode.return_value = True
        mock_rc_instance.get_stats.return_value = {"fallback_mode": True}
        mock_rc.return_value = mock_rc_instance
        
        # Configure SpecializedVerifier mock
        mock_sv_instance = MagicMock()
        mock_sv_instance.verify_text.return_value = [
            Verdict(
                label=VerdictLabel.SUPPORTED,
                confidence=0.9,
                evidence=["Test evidence"],
                reasoning="Test reasoning",
            )
        ]
        mock_sv.return_value = mock_sv_instance
        
        # Configure PromptManager mock
        mock_pm_instance = MagicMock()
        mock_pm.return_value = mock_pm_instance
        
        # Configure JudgeEnsemble mock
        mock_je_instance = MagicMock()
        mock_je_instance.evaluate_all.return_value = [
            JudgeResult(
                model_name="test-judge-1",
                score=85.0,
                reasoning="Test reasoning",
                flagged_issues=[],
                confidence=0.9,
            ),
            JudgeResult(
                model_name="test-judge-2",
                score=80.0,
                reasoning="Test reasoning 2",
                flagged_issues=[],
                confidence=0.85,
            ),
        ]
        mock_je_instance.get_judge_count.return_value = 2
        # Ensure the mock doesn't raise exceptions during initialization
        mock_je.side_effect = None
        mock_je.return_value = mock_je_instance
        
        # Configure AggregationEngine mock
        mock_ae_instance = MagicMock()
        from llm_judge_auditor.models import AggregationMetadata
        mock_ae_instance.aggregate_scores.return_value = (
            82.5,
            AggregationMetadata(
                strategy="mean",
                individual_scores={"test-judge-1": 85.0, "test-judge-2": 80.0},
                variance=12.5,
                is_low_confidence=False,
                weights=None,
            ),
        )
        mock_ae.return_value = mock_ae_instance
        
        # Configure APIKeyManager mock (with API keys available for tests)
        mock_akm_instance = MagicMock()
        mock_akm_instance.groq_key = "mock-groq-key"
        mock_akm_instance.gemini_key = "mock-gemini-key"
        mock_akm_instance.has_any_keys.return_value = True
        mock_akm_instance.load_keys.return_value = {"groq": True, "gemini": True}
        mock_akm_instance.get_setup_instructions.return_value = "Mock setup instructions"
        mock_akm.return_value = mock_akm_instance
        
        # Configure APIJudgeEnsemble mock (used when keys are available)
        mock_aje_instance = MagicMock()
        mock_aje_instance.get_judge_count.return_value = 2
        mock_aje_instance.get_judge_names.return_value = ["groq-llama", "gemini-flash"]
        mock_aje.return_value = mock_aje_instance
        
        yield {
            "model_manager": mock_mm_instance,
            "retrieval": mock_rc_instance,
            "verifier": mock_sv_instance,
            "prompt_manager": mock_pm_instance,
            "judge_ensemble": mock_je_instance,
            "aggregation": mock_ae_instance,
            "api_key_manager": mock_akm_instance,
            "api_judge_ensemble": mock_aje_instance,
        }


class TestEvaluationToolkitInitialization:
    """Test EvaluationToolkit initialization."""

    def test_initialization_success(self, mock_config, mock_components):
        """Test successful initialization of EvaluationToolkit."""
        toolkit = EvaluationToolkit(mock_config)
        
        assert toolkit.config == mock_config
        assert toolkit.model_manager is not None
        assert toolkit.retrieval is not None
        assert toolkit.verifier is not None
        assert toolkit.prompt_manager is not None
        assert toolkit.judge_ensemble is not None
        assert toolkit.aggregation is not None

    def test_initialization_loads_api_keys(self, mock_config, mock_components):
        """Test that initialization loads API keys."""
        toolkit = EvaluationToolkit(mock_config)
        
        # Verify API key manager was initialized
        mock_components["api_key_manager"].load_keys.assert_called_once()

    def test_initialization_failure_raises_error(self, mock_config):
        """Test that initialization failure raises RuntimeError."""
        with patch("llm_judge_auditor.evaluation_toolkit.ModelManager") as mock_mm:
            mock_mm.side_effect = Exception("Model loading failed")
            
            with pytest.raises(RuntimeError, match="Failed to initialize EvaluationToolkit"):
                EvaluationToolkit(mock_config)


class TestEvaluationToolkitEvaluate:
    """Test the main evaluate method."""

    def test_evaluate_basic_flow(self, mock_config, mock_components):
        """Test basic evaluation flow without retrieval."""
        toolkit = EvaluationToolkit(mock_config)
        
        result = toolkit.evaluate(
            source_text="Paris is the capital of France.",
            candidate_output="Paris is the capital of France.",
        )
        
        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.consensus_score == 82.5
        assert len(result.verifier_verdicts) == 1
        assert len(result.judge_results) == 2
        assert result.report is not None

    def test_evaluate_calls_verifier(self, mock_config, mock_components):
        """Test that evaluate calls the specialized verifier."""
        toolkit = EvaluationToolkit(mock_config)
        
        toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
        )
        
        mock_components["verifier"].verify_text.assert_called_once()
        call_args = mock_components["verifier"].verify_text.call_args
        assert call_args[1]["candidate_text"] == "Test candidate"
        assert call_args[1]["source_context"] == "Test source"

    def test_evaluate_calls_judge_ensemble(self, mock_config, mock_components):
        """Test that evaluate calls the judge ensemble."""
        toolkit = EvaluationToolkit(mock_config)
        
        toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
        )
        
        mock_components["judge_ensemble"].evaluate_all.assert_called_once()
        call_args = mock_components["judge_ensemble"].evaluate_all.call_args
        assert call_args[1]["source_text"] == "Test source"
        assert call_args[1]["candidate_output"] == "Test candidate"

    def test_evaluate_calls_aggregation(self, mock_config, mock_components):
        """Test that evaluate calls the aggregation engine."""
        toolkit = EvaluationToolkit(mock_config)
        
        toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
        )
        
        mock_components["aggregation"].aggregate_scores.assert_called_once()
        call_args = mock_components["aggregation"].aggregate_scores.call_args
        assert len(call_args[1]["judge_results"]) == 2
        assert len(call_args[1]["verifier_verdicts"]) == 1

    def test_evaluate_with_retrieval_enabled(self, mock_config, mock_components):
        """Test evaluation with retrieval enabled."""
        # Enable retrieval in config
        mock_config.enable_retrieval = True
        mock_components["retrieval"].fallback_mode.return_value = False
        mock_components["retrieval"].extract_claims.return_value = [
            Mock(text="Test claim", source_span=(0, 10), claim_type="factual")
        ]
        mock_components["retrieval"].retrieve_passages.return_value = [
            Passage(text="Test passage", source="KB:1", relevance_score=0.9)
        ]
        
        toolkit = EvaluationToolkit(mock_config)
        
        result = toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
            use_retrieval=True,
        )
        
        # Verify retrieval was called
        mock_components["retrieval"].extract_claims.assert_called_once()
        mock_components["retrieval"].retrieve_passages.assert_called_once()
        
        # Verify passages were included in report
        assert len(result.report.retrieval_provenance) > 0

    def test_evaluate_empty_source_raises_error(self, mock_config, mock_components):
        """Test that empty source text raises ValueError."""
        toolkit = EvaluationToolkit(mock_config)
        
        with pytest.raises(ValueError, match="source_text cannot be empty"):
            toolkit.evaluate(source_text="", candidate_output="Test")

    def test_evaluate_empty_candidate_raises_error(self, mock_config, mock_components):
        """Test that empty candidate output raises ValueError."""
        toolkit = EvaluationToolkit(mock_config)
        
        with pytest.raises(ValueError, match="candidate_output cannot be empty"):
            toolkit.evaluate(source_text="Test", candidate_output="")

    def test_evaluate_generates_report(self, mock_config, mock_components):
        """Test that evaluate generates a comprehensive report."""
        toolkit = EvaluationToolkit(mock_config)
        
        result = toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
        )
        
        report = result.report
        assert report.metadata is not None
        assert "timestamp" in report.metadata
        assert report.consensus_score == 82.5
        assert len(report.individual_scores) == 2
        assert len(report.reasoning) == 2
        assert report.confidence > 0
        assert report.disagreement_level >= 0

    def test_evaluate_compiles_flagged_issues(self, mock_config, mock_components):
        """Test that evaluate compiles flagged issues from all sources."""
        # Add a refuted verdict
        mock_components["verifier"].verify_text.return_value = [
            Verdict(
                label=VerdictLabel.REFUTED,
                confidence=0.9,
                evidence=["Evidence"],
                reasoning="This is refuted",
            )
        ]
        
        # Add judge issues
        mock_components["judge_ensemble"].evaluate_all.return_value = [
            JudgeResult(
                model_name="test-judge-1",
                score=50.0,
                reasoning="Found issues",
                flagged_issues=[
                    Issue(
                        type=IssueType.HALLUCINATION,
                        severity=IssueSeverity.HIGH,
                        description="Test hallucination",
                        evidence=[],
                    )
                ],
                confidence=0.8,
            ),
        ]
        
        toolkit = EvaluationToolkit(mock_config)
        result = toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
        )
        
        # Should have issues from both verifier and judge
        assert len(result.flagged_issues) >= 2


class TestEvaluationToolkitFactoryMethods:
    """Test factory methods for creating EvaluationToolkit instances."""

    def test_from_preset(self, mock_components):
        """Test creating toolkit from preset."""
        toolkit = EvaluationToolkit.from_preset("fast")
        
        assert toolkit.config is not None
        assert isinstance(toolkit.config, ToolkitConfig)

    def test_from_preset_invalid_raises_error(self, mock_components):
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            EvaluationToolkit.from_preset("invalid_preset")

    def test_from_config_file(self, mock_components, tmp_path):
        """Test creating toolkit from YAML config file."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
verifier_model: "test-verifier"
judge_models:
  - "test-judge-1"
quantize: false
enable_retrieval: false
aggregation_strategy: "mean"
batch_size: 1
max_length: 512
"""
        config_file.write_text(config_content)
        
        toolkit = EvaluationToolkit.from_config_file(str(config_file))
        
        assert toolkit.config is not None
        assert toolkit.config.verifier_model == "test-verifier"


class TestEvaluationToolkitStats:
    """Test statistics and info methods."""

    def test_get_stats(self, mock_config, mock_components):
        """Test getting toolkit statistics."""
        toolkit = EvaluationToolkit(mock_config)
        
        stats = toolkit.get_stats()
        
        assert "config" in stats
        assert "models" in stats
        assert "retrieval" in stats
        assert "num_judges" in stats
        assert stats["num_judges"] == 2


class TestEvaluationToolkitPipelineOrder:
    """Test that the multi-stage pipeline executes in correct order."""

    def test_pipeline_execution_order(self, mock_config, mock_components):
        """Test that pipeline stages execute in the correct order."""
        call_order = []
        
        # Track call order
        mock_components["retrieval"].extract_claims.side_effect = lambda x: (
            call_order.append("retrieval") or []
        )
        mock_components["verifier"].verify_text.side_effect = lambda **kwargs: (
            call_order.append("verifier") or [
                Verdict(VerdictLabel.SUPPORTED, 0.9, [], "")
            ]
        )
        mock_components["judge_ensemble"].evaluate_all.side_effect = lambda **kwargs: (
            call_order.append("ensemble") or [
                JudgeResult("judge1", 80.0, "reasoning", [], 0.9)
            ]
        )
        mock_components["aggregation"].aggregate_scores.side_effect = lambda **kwargs: (
            call_order.append("aggregation") or (
                80.0,
                Mock(
                    strategy="mean",
                    individual_scores={},
                    variance=0,
                    is_low_confidence=False,
                    weights=None,
                ),
            )
        )
        
        toolkit = EvaluationToolkit(mock_config)
        toolkit.evaluate(
            source_text="Test source",
            candidate_output="Test candidate",
        )
        
        # Verify order: verifier -> ensemble -> aggregation
        # (retrieval is skipped because fallback_mode=True)
        assert call_order.index("verifier") < call_order.index("ensemble")
        assert call_order.index("ensemble") < call_order.index("aggregation")


class TestEvaluationToolkitBatchProcessing:
    """Test batch evaluation functionality."""

    def test_batch_evaluate_basic(self, mock_config, mock_components):
        """Test basic batch evaluation with multiple requests."""
        from llm_judge_auditor.models import EvaluationRequest, BatchResult
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text="Paris is the capital of France.",
                candidate_output="Paris is in France.",
            ),
            EvaluationRequest(
                source_text="Water boils at 100°C.",
                candidate_output="Water boils at 100 degrees Celsius.",
            ),
            EvaluationRequest(
                source_text="The Earth orbits the Sun.",
                candidate_output="The Earth goes around the Sun.",
            ),
        ]
        
        batch_result = toolkit.batch_evaluate(requests)
        
        # Verify batch result structure
        assert isinstance(batch_result, BatchResult)
        assert len(batch_result.results) == 3
        assert len(batch_result.errors) == 0
        assert batch_result.metadata["total_requests"] == 3
        assert batch_result.metadata["successful_evaluations"] == 3
        assert batch_result.metadata["failed_evaluations"] == 0
        assert batch_result.metadata["success_rate"] == 1.0

    def test_batch_evaluate_calculates_statistics(self, mock_config, mock_components):
        """Test that batch evaluation calculates correct statistics."""
        from llm_judge_auditor.models import EvaluationRequest
        
        # Mock different scores for each evaluation
        scores = [85.0, 75.0, 90.0, 80.0, 70.0]
        call_count = [0]
        
        def mock_aggregate_with_varying_scores(**kwargs):
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return (
                score,
                Mock(
                    strategy="mean",
                    individual_scores={"judge1": score},
                    variance=5.0,
                    is_low_confidence=False,
                    weights=None,
                ),
            )
        
        mock_components["aggregation"].aggregate_scores.side_effect = mock_aggregate_with_varying_scores
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text=f"Source {i}",
                candidate_output=f"Candidate {i}",
            )
            for i in range(5)
        ]
        
        batch_result = toolkit.batch_evaluate(requests)
        
        # Verify statistics
        stats = batch_result.statistics
        assert stats["count"] == 5
        assert stats["mean"] == 80.0  # (85 + 75 + 90 + 80 + 70) / 5
        assert stats["median"] == 80.0
        assert stats["min"] == 70.0
        assert stats["max"] == 90.0
        assert stats["std"] > 0

    def test_batch_evaluate_error_resilience(self, mock_config, mock_components):
        """Test that batch evaluation continues on error when configured."""
        from llm_judge_auditor.models import EvaluationRequest
        
        # Make the second evaluation fail
        call_count = [0]
        
        def mock_verify_with_failure(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated verification failure")
            return [Verdict(VerdictLabel.SUPPORTED, 0.9, [], "")]
        
        mock_components["verifier"].verify_text.side_effect = mock_verify_with_failure
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text=f"Source {i}",
                candidate_output=f"Candidate {i}",
            )
            for i in range(3)
        ]
        
        batch_result = toolkit.batch_evaluate(requests, continue_on_error=True)
        
        # Should have 2 successful and 1 failed
        assert len(batch_result.results) == 2
        assert len(batch_result.errors) == 1
        assert batch_result.metadata["successful_evaluations"] == 2
        assert batch_result.metadata["failed_evaluations"] == 1
        assert batch_result.metadata["success_rate"] == 2/3
        
        # Check error details
        error = batch_result.errors[0]
        assert error["request_index"] == 1
        assert error["error_type"] == "RuntimeError"
        assert "Simulated verification failure" in error["error_message"]

    def test_batch_evaluate_fails_fast_when_configured(self, mock_config, mock_components):
        """Test that batch evaluation stops on first error when continue_on_error=False."""
        from llm_judge_auditor.models import EvaluationRequest
        
        # Make the second evaluation fail
        call_count = [0]
        
        def mock_verify_with_failure(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated verification failure")
            return [Verdict(VerdictLabel.SUPPORTED, 0.9, [], "")]
        
        mock_components["verifier"].verify_text.side_effect = mock_verify_with_failure
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text=f"Source {i}",
                candidate_output=f"Candidate {i}",
            )
            for i in range(3)
        ]
        
        # Should raise error on second request
        with pytest.raises(RuntimeError, match="Batch evaluation failed at request 2"):
            toolkit.batch_evaluate(requests, continue_on_error=False)

    def test_batch_evaluate_empty_requests_raises_error(self, mock_config, mock_components):
        """Test that empty requests list raises ValueError."""
        toolkit = EvaluationToolkit(mock_config)
        
        with pytest.raises(ValueError, match="requests list cannot be empty"):
            toolkit.batch_evaluate([])

    def test_batch_evaluate_saves_to_json(self, mock_config, mock_components, tmp_path):
        """Test that batch results can be saved to JSON file."""
        from llm_judge_auditor.models import EvaluationRequest
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text="Test source",
                candidate_output="Test candidate",
            ),
        ]
        
        batch_result = toolkit.batch_evaluate(requests)
        
        # Save to file
        output_file = tmp_path / "batch_results.json"
        batch_result.save_to_file(str(output_file))
        
        # Verify file exists and is valid JSON
        assert output_file.exists()
        import json
        with open(output_file) as f:
            data = json.load(f)
        
        assert "results" in data
        assert "errors" in data
        assert "statistics" in data
        assert "metadata" in data

    def test_batch_statistics_with_empty_results(self, mock_config, mock_components):
        """Test that statistics calculation handles empty results gracefully."""
        toolkit = EvaluationToolkit(mock_config)
        
        # Calculate statistics with empty list
        stats = toolkit._calculate_batch_statistics([])
        
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["count"] == 0

    def test_batch_statistics_with_single_result(self, mock_config, mock_components):
        """Test statistics calculation with a single result."""
        from llm_judge_auditor.models import EvaluationRequest
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text="Test source",
                candidate_output="Test candidate",
            ),
        ]
        
        batch_result = toolkit.batch_evaluate(requests)
        
        stats = batch_result.statistics
        assert stats["count"] == 1
        assert stats["mean"] == stats["median"]
        assert stats["min"] == stats["max"]
        assert stats["std"] == 0.0

    def test_batch_evaluate_sequential_processing(self, mock_config, mock_components):
        """Test that batch evaluation processes requests sequentially."""
        from llm_judge_auditor.models import EvaluationRequest
        
        processing_order = []
        
        def track_processing(**kwargs):
            processing_order.append(kwargs["candidate_text"])
            return [Verdict(VerdictLabel.SUPPORTED, 0.9, [], "")]
        
        mock_components["verifier"].verify_text.side_effect = track_processing
        
        toolkit = EvaluationToolkit(mock_config)
        
        requests = [
            EvaluationRequest(
                source_text="Source",
                candidate_output=f"Candidate {i}",
            )
            for i in range(3)
        ]
        
        toolkit.batch_evaluate(requests)
        
        # Verify sequential processing
        assert processing_order == ["Candidate 0", "Candidate 1", "Candidate 2"]

"""
Unit tests for JudgeEnsemble.

Tests the judge ensemble including single evaluation, ensemble evaluation,
pairwise comparison, and response parsing.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from llm_judge_auditor.components.judge_ensemble import (
    BiasDetectionResult,
    JudgeEnsemble,
    PairwiseResult,
)
from llm_judge_auditor.components.model_manager import ModelManager
from llm_judge_auditor.components.prompt_manager import PromptManager
from llm_judge_auditor.models import IssueType, IssueSeverity, JudgeResult


class TestJudgeEnsemble:
    """Test suite for JudgeEnsemble class."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager with loaded judges."""
        mock_mm = Mock(spec=ModelManager)
        
        # Create mock models and tokenizers
        mock_model1 = Mock()
        mock_model1.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model1.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer1 = Mock()
        mock_tokenizer1.pad_token_id = 0
        mock_tokenizer1.eos_token_id = 1
        mock_tokenizer1.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer1.decode = Mock(return_value="REASONING: Test reasoning\nSCORE: 75\nFLAGGED_ISSUES: None detected")
        
        mock_model2 = Mock()
        mock_model2.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model2.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer2 = Mock()
        mock_tokenizer2.pad_token_id = 0
        mock_tokenizer2.eos_token_id = 1
        mock_tokenizer2.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer2.decode = Mock(return_value="REASONING: Test reasoning\nSCORE: 80\nFLAGGED_ISSUES: None detected")
        
        # Setup get_judge to return appropriate mocks
        def get_judge_side_effect(name):
            if name == "judge1":
                return (mock_model1, mock_tokenizer1)
            elif name == "judge2":
                return (mock_model2, mock_tokenizer2)
            return None
        
        mock_mm.get_judge = Mock(side_effect=get_judge_side_effect)
        mock_mm.get_all_judges = Mock(return_value={
            "judge1": (mock_model1, mock_tokenizer1),
            "judge2": (mock_model2, mock_tokenizer2),
        })
        
        return mock_mm

    @pytest.fixture
    def mock_prompt_manager(self):
        """Create a mock PromptManager."""
        mock_pm = Mock(spec=PromptManager)
        mock_pm.get_prompt = Mock(return_value="Test prompt")
        return mock_pm

    def test_init(self, mock_model_manager, mock_prompt_manager):
        """Test JudgeEnsemble initialization."""
        ensemble = JudgeEnsemble(
            model_manager=mock_model_manager,
            prompt_manager=mock_prompt_manager,
            max_length=1024,
            temperature=0.2,
        )
        
        assert ensemble.model_manager == mock_model_manager
        assert ensemble.prompt_manager == mock_prompt_manager
        assert ensemble.max_length == 1024
        assert ensemble.temperature == 0.2

    def test_init_default_prompt_manager(self, mock_model_manager):
        """Test initialization with default PromptManager."""
        ensemble = JudgeEnsemble(model_manager=mock_model_manager)
        
        assert ensemble.prompt_manager is not None
        assert isinstance(ensemble.prompt_manager, PromptManager)

    def test_generate_response(self, mock_model_manager, mock_prompt_manager):
        """Test response generation from a judge model."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode = Mock(return_value="Generated response")
        
        response = ensemble._generate_response(mock_model, mock_tokenizer, "Test prompt")
        
        assert response == "Generated response"
        mock_tokenizer.assert_called_once()
        mock_model.generate.assert_called_once()

    def test_parse_factual_accuracy_response_valid(self, mock_model_manager, mock_prompt_manager):
        """Test parsing a valid factual accuracy response."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: The candidate output contains accurate information about Paris.
All claims are supported by the source text.

SCORE: 85

FLAGGED_ISSUES: None detected
"""
        
        result = ensemble._parse_factual_accuracy_response(response, "test-judge")
        
        assert isinstance(result, JudgeResult)
        assert result.model_name == "test-judge"
        assert result.score == 85.0
        assert "accurate information" in result.reasoning
        assert len(result.flagged_issues) == 0

    def test_parse_factual_accuracy_response_with_issues(self, mock_model_manager, mock_prompt_manager):
        """Test parsing response with flagged issues."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: The candidate contains some inaccuracies.

SCORE: 45

FLAGGED_ISSUES:
Paris is not the capital of Germany
The date is incorrect
"""
        
        result = ensemble._parse_factual_accuracy_response(response, "test-judge")
        
        assert result.score == 45.0
        assert len(result.flagged_issues) == 2
        assert result.flagged_issues[0].type == IssueType.HALLUCINATION
        assert result.flagged_issues[0].severity == IssueSeverity.MEDIUM

    def test_parse_factual_accuracy_response_score_clamping(self, mock_model_manager, mock_prompt_manager):
        """Test that scores are clamped to 0-100 range."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        # Test score > 100
        response1 = "REASONING: Test\nSCORE: 150\nFLAGGED_ISSUES: None"
        result1 = ensemble._parse_factual_accuracy_response(response1, "test-judge")
        assert result1.score == 100.0
        
        # Test score < 0
        response2 = "REASONING: Test\nSCORE: -10\nFLAGGED_ISSUES: None"
        result2 = ensemble._parse_factual_accuracy_response(response2, "test-judge")
        assert result2.score == 0.0

    def test_parse_factual_accuracy_response_missing_score(self, mock_model_manager, mock_prompt_manager):
        """Test parsing response with missing score."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = "REASONING: Some reasoning without a score"
        
        result = ensemble._parse_factual_accuracy_response(response, "test-judge")
        
        # Should default to 50
        assert result.score == 50.0
        # Should strip REASONING: prefix
        assert result.reasoning == "Some reasoning without a score"

    def test_parse_pairwise_response_valid(self, mock_model_manager, mock_prompt_manager):
        """Test parsing a valid pairwise comparison response."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: Candidate A is more accurate than Candidate B.
Candidate A has no hallucinations while B contains false claims.

WINNER: A

EXPLANATION: A is clearly superior in factual accuracy.
"""
        
        result = ensemble._parse_pairwise_response(response)
        
        assert isinstance(result, PairwiseResult)
        assert result.winner == "A"
        assert "more accurate" in result.reasoning
        assert "clearly superior" in result.reasoning

    def test_parse_pairwise_response_tie(self, mock_model_manager, mock_prompt_manager):
        """Test parsing pairwise response with tie."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = "REASONING: Both are equivalent\nWINNER: TIE"
        
        result = ensemble._parse_pairwise_response(response)
        
        assert result.winner == "TIE"

    def test_parse_pairwise_response_missing_winner(self, mock_model_manager, mock_prompt_manager):
        """Test parsing pairwise response with missing winner."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = "REASONING: Some comparison without a clear winner"
        
        result = ensemble._parse_pairwise_response(response)
        
        # Should default to TIE
        assert result.winner == "TIE"

    def test_evaluate_single_success(self, mock_model_manager, mock_prompt_manager):
        """Test successful single judge evaluation."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        result = ensemble.evaluate_single(
            judge_name="judge1",
            source_text="Paris is the capital of France.",
            candidate_output="Paris is the capital of France.",
        )
        
        assert isinstance(result, JudgeResult)
        assert result.model_name == "judge1"
        assert 0 <= result.score <= 100
        mock_prompt_manager.get_prompt.assert_called_once()

    def test_evaluate_single_with_retrieved_context(self, mock_model_manager, mock_prompt_manager):
        """Test single evaluation with retrieved context."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        result = ensemble.evaluate_single(
            judge_name="judge1",
            source_text="Paris is the capital of France.",
            candidate_output="Paris is the capital of France.",
            retrieved_context="Additional context from Wikipedia.",
        )
        
        assert isinstance(result, JudgeResult)
        # Check that retrieved_context was passed to prompt
        call_kwargs = mock_prompt_manager.get_prompt.call_args[1]
        assert call_kwargs["retrieved_context"] == "Additional context from Wikipedia."

    def test_evaluate_single_judge_not_loaded(self, mock_model_manager, mock_prompt_manager):
        """Test evaluation with unloaded judge."""
        mock_model_manager.get_judge = Mock(return_value=None)
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        with pytest.raises(ValueError, match="not loaded"):
            ensemble.evaluate_single(
                judge_name="nonexistent",
                source_text="Test",
                candidate_output="Test",
            )

    def test_evaluate_all_success(self, mock_model_manager, mock_prompt_manager):
        """Test successful ensemble evaluation with all judges."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        results = ensemble.evaluate_all(
            source_text="Paris is the capital of France.",
            candidate_output="Paris is the capital of France.",
        )
        
        assert len(results) == 2
        assert all(isinstance(r, JudgeResult) for r in results)
        assert results[0].model_name == "judge1"
        assert results[1].model_name == "judge2"

    def test_evaluate_all_no_judges_loaded(self, mock_model_manager, mock_prompt_manager):
        """Test ensemble evaluation with no judges loaded."""
        mock_model_manager.get_all_judges = Mock(return_value={})
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        with pytest.raises(RuntimeError, match="No judge models loaded"):
            ensemble.evaluate_all(
                source_text="Test",
                candidate_output="Test",
            )

    def test_evaluate_all_partial_failure(self, mock_model_manager, mock_prompt_manager):
        """Test ensemble evaluation with partial failures."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        # Make judge2 fail
        original_get_judge = mock_model_manager.get_judge
        def failing_get_judge(name):
            if name == "judge2":
                raise RuntimeError("Judge2 failed")
            return original_get_judge(name)
        
        mock_model_manager.get_judge = Mock(side_effect=failing_get_judge)
        
        results = ensemble.evaluate_all(
            source_text="Test",
            candidate_output="Test",
        )
        
        # Should have one successful result
        assert len(results) == 1
        assert results[0].model_name == "judge1"

    def test_evaluate_all_complete_failure(self, mock_model_manager, mock_prompt_manager):
        """Test ensemble evaluation when all judges fail."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        # Make all judges fail
        mock_model_manager.get_judge = Mock(side_effect=RuntimeError("All judges failed"))
        
        with pytest.raises(RuntimeError, match="All judge evaluations failed"):
            ensemble.evaluate_all(
                source_text="Test",
                candidate_output="Test",
            )

    def test_pairwise_compare_success(self, mock_model_manager, mock_prompt_manager):
        """Test successful pairwise comparison."""
        # Setup mock to return pairwise response
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode = Mock(return_value="REASONING: A is better\nWINNER: A")
        
        mock_model_manager.get_judge = Mock(return_value=(mock_model, mock_tokenizer))
        
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        result = ensemble.pairwise_compare(
            source_text="Paris is the capital of France.",
            candidate_a="Paris is the capital of France.",
            candidate_b="Paris is the capital of Germany.",
            judge_name="judge1",
        )
        
        assert isinstance(result, PairwiseResult)
        assert result.winner in ["A", "B", "TIE"]
        assert result.reasoning is not None
        mock_prompt_manager.get_prompt.assert_called_once()

    def test_pairwise_compare_default_judge(self, mock_model_manager, mock_prompt_manager):
        """Test pairwise comparison with default judge selection."""
        # Setup mock to return pairwise response
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode = Mock(return_value="REASONING: Tie\nWINNER: TIE")
        
        mock_model_manager.get_all_judges = Mock(return_value={
            "judge1": (mock_model, mock_tokenizer)
        })
        
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        result = ensemble.pairwise_compare(
            source_text="Test",
            candidate_a="Test A",
            candidate_b="Test B",
        )
        
        assert isinstance(result, PairwiseResult)

    def test_pairwise_compare_no_judges(self, mock_model_manager, mock_prompt_manager):
        """Test pairwise comparison with no judges loaded."""
        mock_model_manager.get_all_judges = Mock(return_value={})
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        with pytest.raises(ValueError, match="No judge models loaded"):
            ensemble.pairwise_compare(
                source_text="Test",
                candidate_a="Test A",
                candidate_b="Test B",
            )

    def test_pairwise_compare_invalid_judge(self, mock_model_manager, mock_prompt_manager):
        """Test pairwise comparison with invalid judge name."""
        mock_model_manager.get_judge = Mock(return_value=None)
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        with pytest.raises(ValueError, match="not loaded"):
            ensemble.pairwise_compare(
                source_text="Test",
                candidate_a="Test A",
                candidate_b="Test B",
                judge_name="nonexistent",
            )

    def test_confidence_calculation(self, mock_model_manager, mock_prompt_manager):
        """Test that confidence is calculated based on score extremity."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        # High score should have high confidence
        response_high = "REASONING: Test\nSCORE: 95\nFLAGGED_ISSUES: None"
        result_high = ensemble._parse_factual_accuracy_response(response_high, "test")
        assert result_high.confidence > 0.8
        
        # Low score should have high confidence
        response_low = "REASONING: Test\nSCORE: 5\nFLAGGED_ISSUES: None"
        result_low = ensemble._parse_factual_accuracy_response(response_low, "test")
        assert result_low.confidence > 0.8
        
        # Middle score should have low confidence
        response_mid = "REASONING: Test\nSCORE: 50\nFLAGGED_ISSUES: None"
        result_mid = ensemble._parse_factual_accuracy_response(response_mid, "test")
        assert result_mid.confidence == 0.0

    def test_parse_bias_detection_response_no_bias(self, mock_model_manager, mock_prompt_manager):
        """Test parsing bias detection response with no bias detected."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: The text was analyzed for bias and stereotypes.
No problematic language was found.

FLAGGED_PHRASES: None detected

OVERALL_ASSESSMENT: No bias detected in the candidate output.
"""
        
        result = ensemble._parse_bias_detection_response(response, "test-judge")
        
        assert isinstance(result, BiasDetectionResult)
        assert result.model_name == "test-judge"
        assert len(result.flagged_phrases) == 0
        assert "No bias detected" in result.overall_assessment
        assert "analyzed for bias" in result.reasoning

    def test_parse_bias_detection_response_with_bias(self, mock_model_manager, mock_prompt_manager):
        """Test parsing bias detection response with flagged phrases."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: The text contains gender stereotypes.

FLAGGED_PHRASES:
- "Women are naturally better at nursing" - Gender stereotype [SEVERITY: high]
- "Men are not emotional" - Gender stereotype [SEVERITY: medium]

OVERALL_ASSESSMENT: Detected 2 instances of gender bias.
"""
        
        result = ensemble._parse_bias_detection_response(response, "test-judge")
        
        assert len(result.flagged_phrases) == 2
        assert all(issue.type == IssueType.BIAS for issue in result.flagged_phrases)
        assert result.flagged_phrases[0].severity == IssueSeverity.HIGH
        assert result.flagged_phrases[1].severity == IssueSeverity.MEDIUM
        assert "gender bias" in result.overall_assessment.lower()

    def test_parse_bias_detection_response_quoted_phrases(self, mock_model_manager, mock_prompt_manager):
        """Test parsing bias detection with quoted phrase format."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: Found stereotypical language.

FLAGGED_PHRASES:
"elderly people are slow" - Age-based stereotype [SEVERITY: low]
"all teenagers are irresponsible" - Age-based generalization [SEVERITY: medium]

OVERALL_ASSESSMENT: Age-related bias detected.
"""
        
        result = ensemble._parse_bias_detection_response(response, "test-judge")
        
        assert len(result.flagged_phrases) == 2
        assert "elderly people are slow" in result.flagged_phrases[0].description
        assert result.flagged_phrases[0].severity == IssueSeverity.LOW
        assert result.flagged_phrases[1].severity == IssueSeverity.MEDIUM

    def test_parse_bias_detection_response_default_severity(self, mock_model_manager, mock_prompt_manager):
        """Test that missing severity defaults to MEDIUM."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: Found bias.

FLAGGED_PHRASES:
- Problematic phrase without severity rating

OVERALL_ASSESSMENT: Bias detected.
"""
        
        result = ensemble._parse_bias_detection_response(response, "test-judge")
        
        assert len(result.flagged_phrases) == 1
        assert result.flagged_phrases[0].severity == IssueSeverity.MEDIUM

    def test_parse_bias_detection_response_default_assessment(self, mock_model_manager, mock_prompt_manager):
        """Test that missing overall assessment is generated from flagged phrases."""
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        response = """
REASONING: Analysis complete.

FLAGGED_PHRASES:
- "phrase 1" - explanation [SEVERITY: high]
- "phrase 2" - explanation [SEVERITY: medium]
- "phrase 3" - explanation [SEVERITY: low]
"""
        
        result = ensemble._parse_bias_detection_response(response, "test-judge")
        
        assert len(result.flagged_phrases) == 3
        # Should generate default assessment
        assert "3 instances of bias" in result.overall_assessment
        assert "1 high severity" in result.overall_assessment
        assert "1 medium severity" in result.overall_assessment
        assert "1 low severity" in result.overall_assessment

    def test_detect_bias_success(self, mock_model_manager, mock_prompt_manager):
        """Test successful bias detection."""
        # Setup mock to return bias detection response
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode = Mock(
            return_value='REASONING: Found bias\nFLAGGED_PHRASES:\n"biased phrase" - explanation [SEVERITY: high]\nOVERALL_ASSESSMENT: Bias detected'
        )
        
        mock_model_manager.get_judge = Mock(return_value=(mock_model, mock_tokenizer))
        
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        result = ensemble.detect_bias(
            candidate_output="Text with potential bias.",
            judge_name="judge1",
        )
        
        assert isinstance(result, BiasDetectionResult)
        assert result.model_name == "judge1"
        assert len(result.flagged_phrases) >= 0
        assert result.overall_assessment is not None
        mock_prompt_manager.get_prompt.assert_called_once_with(
            task="bias_detection",
            candidate_output="Text with potential bias.",
        )

    def test_detect_bias_default_judge(self, mock_model_manager, mock_prompt_manager):
        """Test bias detection with default judge selection."""
        # Setup mock to return bias detection response
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=[torch.zeros(1)])
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode = Mock(
            return_value="REASONING: No bias\nFLAGGED_PHRASES: None\nOVERALL_ASSESSMENT: Clean"
        )
        
        mock_model_manager.get_all_judges = Mock(return_value={
            "judge1": (mock_model, mock_tokenizer)
        })
        
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        result = ensemble.detect_bias(candidate_output="Clean text.")
        
        assert isinstance(result, BiasDetectionResult)

    def test_detect_bias_no_judges(self, mock_model_manager, mock_prompt_manager):
        """Test bias detection with no judges loaded."""
        mock_model_manager.get_all_judges = Mock(return_value={})
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        with pytest.raises(ValueError, match="No judge models loaded"):
            ensemble.detect_bias(candidate_output="Test text")

    def test_detect_bias_invalid_judge(self, mock_model_manager, mock_prompt_manager):
        """Test bias detection with invalid judge name."""
        mock_model_manager.get_judge = Mock(return_value=None)
        ensemble = JudgeEnsemble(mock_model_manager, mock_prompt_manager)
        
        with pytest.raises(ValueError, match="not loaded"):
            ensemble.detect_bias(
                candidate_output="Test text",
                judge_name="nonexistent",
            )

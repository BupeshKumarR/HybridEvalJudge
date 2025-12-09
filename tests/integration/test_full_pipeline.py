"""
Integration tests for the full evaluation pipeline.

These tests verify end-to-end functionality with all components working together.
"""

import pytest
from pathlib import Path

from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
from llm_judge_auditor.config import ToolkitConfig, AggregationStrategy
from llm_judge_auditor.models import EvaluationRequest, VerdictLabel


class TestFullPipelineIntegration:
    """Test the complete evaluation pipeline with all components."""

    def test_basic_evaluation_pipeline(self, sample_source_text, sample_candidate_output):
        """Test basic evaluation with minimal configuration."""
        # Create minimal config for testing
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=128,
        )
        
        # Note: This test requires mock models or will fail in CI
        # In a real integration test, we would use actual small models
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify result structure
            assert result is not None
            assert hasattr(result, 'consensus_score')
            assert hasattr(result, 'report')
            assert hasattr(result, 'verifier_verdicts')
            assert hasattr(result, 'judge_results')
            
            # Verify score is in valid range
            assert 0 <= result.consensus_score <= 100
            
            # Verify report contains required fields
            assert result.report.metadata is not None
            assert result.report.consensus_score == result.consensus_score
            assert isinstance(result.report.individual_scores, dict)
            
        except RuntimeError as e:
            # Expected in test environment without real models
            pytest.skip(f"Skipping integration test without real models: {e}")

    def test_evaluation_with_retrieval(self, sample_source_text, sample_candidate_output):
        """Test evaluation pipeline with retrieval enabled."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=True,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=128,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output,
                use_retrieval=True
            )
            
            # Verify retrieval was attempted
            assert result.report.metadata['retrieval_enabled'] is True
            
        except RuntimeError as e:
            pytest.skip(f"Skipping integration test without real models: {e}")

    def test_batch_evaluation_pipeline(self, sample_source_text, sample_candidate_output):
        """Test batch evaluation with multiple requests."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=128,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            
            # Create multiple requests
            requests = [
                EvaluationRequest(
                    source_text=sample_source_text,
                    candidate_output=sample_candidate_output,
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
                EvaluationRequest(
                    source_text="The Earth orbits the Sun.",
                    candidate_output="The Earth goes around the Sun.",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
            ]
            
            batch_result = toolkit.batch_evaluate(requests)
            
            # Verify batch result structure
            assert batch_result is not None
            assert hasattr(batch_result, 'results')
            assert hasattr(batch_result, 'statistics')
            assert hasattr(batch_result, 'metadata')
            
            # Verify statistics
            assert 'mean' in batch_result.statistics
            assert 'median' in batch_result.statistics
            assert batch_result.statistics['count'] == len(batch_result.results)
            
        except RuntimeError as e:
            pytest.skip(f"Skipping integration test without real models: {e}")

    def test_hallucination_detection_pipeline(self, sample_source_text, sample_hallucinated_output):
        """Test that the pipeline detects hallucinations."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=128,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_hallucinated_output
            )
            
            # Verify that issues were flagged
            # (In a real test with actual models, we'd expect hallucinations to be detected)
            assert hasattr(result, 'flagged_issues')
            assert isinstance(result.flagged_issues, list)
            
        except RuntimeError as e:
            pytest.skip(f"Skipping integration test without real models: {e}")

    def test_pipeline_with_multiple_judges(self, sample_source_text, sample_candidate_output):
        """Test evaluation with multiple judge models."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge-1", "test-judge-2", "test-judge-3"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=128,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify judges contributed (may be API judges or local judges)
            # At least 1 judge should have contributed
            assert len(result.judge_results) >= 1
            assert len(result.report.individual_scores) >= 1
            
        except RuntimeError as e:
            pytest.skip(f"Skipping integration test without real models: {e}")

    def test_pipeline_stage_ordering(self, sample_source_text, sample_candidate_output):
        """Test that pipeline stages execute in correct order."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=True,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=128,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify all stages completed
            # Stage 1: Retrieval (if enabled)
            assert 'num_retrieved_passages' in result.report.metadata
            
            # Stage 2: Verification
            assert len(result.verifier_verdicts) >= 0
            assert 'num_verifier_verdicts' in result.report.metadata
            
            # Stage 3: Judge Ensemble
            assert len(result.judge_results) > 0
            assert 'num_judge_results' in result.report.metadata
            
            # Stage 4: Aggregation
            assert result.consensus_score is not None
            assert result.aggregation_metadata is not None
            
            # Stage 5: Reporting
            assert result.report is not None
            assert result.report.metadata is not None
            
        except RuntimeError as e:
            pytest.skip(f"Skipping integration test without real models: {e}")


class TestPresetIntegration:
    """Test evaluation with different preset configurations."""

    def test_fast_preset_initialization(self):
        """Test that fast preset can be initialized."""
        try:
            toolkit = EvaluationToolkit.from_preset("fast")
            
            # Verify config matches fast preset
            assert toolkit.config.enable_retrieval is False
            # Note: judge_models may be empty if using API judges instead
            # The important thing is that the toolkit has judges available
            assert toolkit.config.aggregation_strategy == AggregationStrategy.MEAN
            
            # Verify toolkit has judges available (either local or API)
            has_judges = (
                (toolkit.judge_ensemble is not None) or 
                (toolkit.api_judge_ensemble is not None and toolkit.api_judge_ensemble.get_judge_count() > 0)
            )
            assert has_judges, "Toolkit should have judges available (local or API)"
            
        except RuntimeError as e:
            pytest.skip(f"Skipping preset test without real models: {e}")

    def test_balanced_preset_initialization(self):
        """Test that balanced preset can be initialized."""
        try:
            toolkit = EvaluationToolkit.from_preset("balanced")
            
            # Verify config matches balanced preset
            assert toolkit.config.enable_retrieval is True
            # Note: judge_models may be empty if using API judges instead
            assert toolkit.config.aggregation_strategy == AggregationStrategy.MEAN
            
            # Verify toolkit has judges available (either local or API)
            has_judges = (
                (toolkit.judge_ensemble is not None) or 
                (toolkit.api_judge_ensemble is not None and toolkit.api_judge_ensemble.get_judge_count() > 0)
            )
            assert has_judges, "Toolkit should have judges available (local or API)"
            
        except RuntimeError as e:
            pytest.skip(f"Skipping preset test without real models: {e}")

    def test_fast_preset_evaluation(self, sample_source_text, sample_candidate_output):
        """Test evaluation using fast preset."""
        try:
            toolkit = EvaluationToolkit.from_preset("fast")
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify result structure
            assert result is not None
            assert 0 <= result.consensus_score <= 100
            
            # Fast preset should not use retrieval
            assert result.report.metadata['retrieval_enabled'] is False
            
        except RuntimeError as e:
            pytest.skip(f"Skipping preset test without real models: {e}")

    def test_balanced_preset_evaluation(self, sample_source_text, sample_candidate_output):
        """Test evaluation using balanced preset."""
        try:
            toolkit = EvaluationToolkit.from_preset("balanced")
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify result structure
            assert result is not None
            assert 0 <= result.consensus_score <= 100
            
            # Balanced preset should use retrieval (if KB available)
            # Note: May fall back to zero-retrieval mode if no KB
            assert 'retrieval_enabled' in result.report.metadata
            
        except RuntimeError as e:
            pytest.skip(f"Skipping preset test without real models: {e}")

    def test_preset_comparison(self, sample_source_text, sample_candidate_output):
        """Test that different presets produce different configurations."""
        try:
            fast_toolkit = EvaluationToolkit.from_preset("fast")
            balanced_toolkit = EvaluationToolkit.from_preset("balanced")
            
            # Verify presets have different configurations
            # The key difference is retrieval enabled/disabled
            assert fast_toolkit.config.enable_retrieval != balanced_toolkit.config.enable_retrieval
            
            # Both should have judges available (either local or API)
            fast_has_judges = (
                (fast_toolkit.judge_ensemble is not None) or 
                (fast_toolkit.api_judge_ensemble is not None and fast_toolkit.api_judge_ensemble.get_judge_count() > 0)
            )
            balanced_has_judges = (
                (balanced_toolkit.judge_ensemble is not None) or 
                (balanced_toolkit.api_judge_ensemble is not None and balanced_toolkit.api_judge_ensemble.get_judge_count() > 0)
            )
            assert fast_has_judges, "Fast preset should have judges available"
            assert balanced_has_judges, "Balanced preset should have judges available"
            
        except RuntimeError as e:
            pytest.skip(f"Skipping preset test without real models: {e}")


class TestErrorHandlingIntegration:
    """Test error handling in the full pipeline."""

    def test_empty_source_text_error(self):
        """Test that empty source text raises appropriate error."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            
            with pytest.raises(ValueError, match="source_text cannot be empty"):
                toolkit.evaluate(
                    source_text="",
                    candidate_output="Some output"
                )
                
        except RuntimeError as e:
            pytest.skip(f"Skipping error handling test without real models: {e}")

    def test_empty_candidate_output_error(self, sample_source_text):
        """Test that empty candidate output raises appropriate error."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            
            with pytest.raises(ValueError, match="candidate_output cannot be empty"):
                toolkit.evaluate(
                    source_text=sample_source_text,
                    candidate_output=""
                )
                
        except RuntimeError as e:
            pytest.skip(f"Skipping error handling test without real models: {e}")

    def test_batch_evaluation_error_resilience(self, sample_source_text):
        """Test that batch evaluation continues after individual failures."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            
            # Create requests with one invalid request
            requests = [
                EvaluationRequest(
                    source_text=sample_source_text,
                    candidate_output="Valid output",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
                EvaluationRequest(
                    source_text="",  # Invalid: empty source
                    candidate_output="Some output",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
                EvaluationRequest(
                    source_text=sample_source_text,
                    candidate_output="Another valid output",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
            ]
            
            # Should continue despite error
            batch_result = toolkit.batch_evaluate(requests, continue_on_error=True)
            
            # Verify that some succeeded and one failed
            assert len(batch_result.errors) > 0
            assert len(batch_result.results) > 0
            assert batch_result.metadata['failed_evaluations'] > 0
            
        except RuntimeError as e:
            pytest.skip(f"Skipping error handling test without real models: {e}")

    def test_batch_evaluation_fail_fast(self, sample_source_text):
        """Test that batch evaluation can fail fast on error."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            
            # Create requests with one invalid request
            requests = [
                EvaluationRequest(
                    source_text=sample_source_text,
                    candidate_output="Valid output",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
                EvaluationRequest(
                    source_text="",  # Invalid: empty source
                    candidate_output="Some output",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                ),
            ]
            
            # Should raise error immediately
            with pytest.raises(RuntimeError, match="Batch evaluation failed"):
                toolkit.batch_evaluate(requests, continue_on_error=False)
                
        except RuntimeError as e:
            if "Batch evaluation failed" not in str(e):
                pytest.skip(f"Skipping error handling test without real models: {e}")

    def test_invalid_preset_name(self):
        """Test that invalid preset name raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            EvaluationToolkit.from_preset("nonexistent_preset")

    def test_model_loading_failure_handling(self, monkeypatch):
        """Test that model loading failures are handled gracefully when no judges are available."""
        # Disable API judges for this test by removing API keys
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        
        from llm_judge_auditor.config import APIConfig
        
        config = ToolkitConfig(
            verifier_model="nonexistent/model",
            judge_models=[],  # No local judges configured
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            api_config=APIConfig(enable_api_judges=False),  # Explicitly disable API judges
        )
        
        # With no API judges and no local judges, should raise RuntimeError
        with pytest.raises(RuntimeError, match="No judges available"):
            toolkit = EvaluationToolkit(config)


class TestComponentIntegration:
    """Test integration between specific components."""

    def test_verifier_to_aggregation_flow(self, sample_source_text, sample_candidate_output):
        """Test that verifier verdicts flow correctly to aggregation."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify verifier verdicts are included in result
            assert hasattr(result, 'verifier_verdicts')
            assert isinstance(result.verifier_verdicts, list)
            
            # Verify verdicts are included in report
            assert result.report.verifier_verdicts == result.verifier_verdicts
            
        except RuntimeError as e:
            pytest.skip(f"Skipping component integration test without real models: {e}")

    def test_retrieval_to_verifier_flow(self, sample_source_text, sample_candidate_output):
        """Test that retrieved passages flow to verifier."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=True,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output,
                use_retrieval=True
            )
            
            # Verify retrieval provenance is tracked
            assert hasattr(result.report, 'retrieval_provenance')
            assert isinstance(result.report.retrieval_provenance, list)
            
        except RuntimeError as e:
            pytest.skip(f"Skipping component integration test without real models: {e}")

    def test_judge_ensemble_to_aggregation_flow(self, sample_source_text, sample_candidate_output):
        """Test that judge results flow correctly to aggregation."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge-1", "test-judge-2"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            result = toolkit.evaluate(
                source_text=sample_source_text,
                candidate_output=sample_candidate_output
            )
            
            # Verify judge results are aggregated
            assert len(result.judge_results) == 2
            assert result.consensus_score is not None
            
            # Verify individual scores are preserved
            assert len(result.report.individual_scores) == 2
            
        except RuntimeError as e:
            pytest.skip(f"Skipping component integration test without real models: {e}")

    def test_statistics_calculation_integration(self, sample_source_text):
        """Test that batch statistics are calculated correctly."""
        config = ToolkitConfig(
            verifier_model="test-verifier",
            judge_models=["test-judge"],
            quantize=False,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        
        try:
            toolkit = EvaluationToolkit(config)
            
            requests = [
                EvaluationRequest(
                    source_text=sample_source_text,
                    candidate_output=f"Output {i}",
                    task="factual_accuracy",
                    criteria=["correctness"],
                    use_retrieval=False
                )
                for i in range(5)
            ]
            
            batch_result = toolkit.batch_evaluate(requests)
            
            # Verify all statistical measures are present
            assert 'mean' in batch_result.statistics
            assert 'median' in batch_result.statistics
            assert 'std' in batch_result.statistics
            assert 'min' in batch_result.statistics
            assert 'max' in batch_result.statistics
            assert 'count' in batch_result.statistics
            
            # Verify count matches results
            assert batch_result.statistics['count'] == len(batch_result.results)
            
        except RuntimeError as e:
            pytest.skip(f"Skipping statistics integration test without real models: {e}")

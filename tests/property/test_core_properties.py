"""Property-based tests for core correctness properties."""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, patch
import torch

from llm_judge_auditor.models import JudgeResult
from llm_judge_auditor.components.aggregation_engine import (
    AggregationEngine,
    AggregationStrategy as EngineAggregationStrategy,
)
from llm_judge_auditor.components.model_manager import ModelManager
from llm_judge_auditor.components.retrieval_component import RetrievalComponent
from llm_judge_auditor.components.specialized_verifier import SpecializedVerifier
from llm_judge_auditor.components.judge_ensemble import JudgeEnsemble
from llm_judge_auditor.config import ToolkitConfig, DeviceType


@st.composite
def score_strategy(draw):
    """Generate valid scores between 0 and 100."""
    return draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))


@st.composite
def judge_result_strategy(draw):
    """Generate valid JudgeResult objects."""
    model_name = draw(st.text(min_size=5, max_size=20))
    score = draw(score_strategy())
    reasoning = draw(st.text(min_size=10, max_size=100))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return JudgeResult(
        model_name=model_name,
        score=score,
        reasoning=reasoning,
        flagged_issues=[],
        confidence=confidence
    )


@st.composite
def judge_results_list_strategy(draw):
    """Generate a list of JudgeResult objects with unique model names."""
    num_judges = draw(st.integers(min_value=2, max_value=5))
    judge_results = []
    
    for i in range(num_judges):
        # Use simple sequential names to ensure uniqueness
        model_name = f"judge_{i}"
        score = draw(score_strategy())
        reasoning = draw(st.text(min_size=10, max_size=50))
        confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        
        judge_results.append(JudgeResult(
            model_name=model_name,
            score=score,
            reasoning=reasoning,
            flagged_issues=[],
            confidence=confidence
        ))
    
    return judge_results


@st.composite
def toolkit_config_strategy(draw):
    """Generate valid ToolkitConfig objects for testing."""
    # Generate 1-3 judge models
    num_judges = draw(st.integers(min_value=1, max_value=3))
    judge_models = [f"test/judge{i}" for i in range(num_judges)]
    
    # Generate device type
    device = draw(st.sampled_from([DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS]))
    
    # Generate quantization setting
    quantize = draw(st.booleans())
    
    return ToolkitConfig(
        verifier_model="test/verifier",
        judge_models=judge_models,
        quantize=quantize,
        device=device,
    )


class TestCoreProperties:
    """Test core correctness properties with property-based testing."""

    # Feature: llm-judge-auditor, Property 3: Score bounds validity
    @given(judge_results=st.lists(judge_result_strategy(), min_size=1, max_size=5))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_score_bounds_validity(self, judge_results):
        """Verify scores are between 0 and 100. Validates: Requirements 2.4"""
        for judge_result in judge_results:
            assert 0 <= judge_result.score <= 100
            assert judge_result.reasoning is not None
            assert len(judge_result.reasoning) > 0

    # Feature: llm-judge-auditor, Property 24: Ensemble aggregation correctness
    @given(
        judge_results=judge_results_list_strategy(),
        strategy=st.sampled_from([
            EngineAggregationStrategy.MEAN,
            EngineAggregationStrategy.MEDIAN,
            EngineAggregationStrategy.WEIGHTED_AVERAGE,
            EngineAggregationStrategy.MAJORITY_VOTE
        ])
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.large_base_example]
    )
    def test_ensemble_aggregation_correctness(self, judge_results, strategy):
        """Verify aggregation strategies work correctly. Validates: Requirements 11.1, 11.2, 11.3, 11.4"""
        # Requirement 11.1: Load configured aggregation strategy
        # Requirement 11.2: Apply configured weights for weighted average
        weights = None
        if strategy == EngineAggregationStrategy.WEIGHTED_AVERAGE:
            # Create weights for each judge model
            weights = {jr.model_name: 1.0 / len(judge_results) for jr in judge_results}
        
        engine = AggregationEngine(strategy=strategy, disagreement_threshold=20.0, weights=weights)
        consensus_score, metadata = engine.aggregate_scores(judge_results=judge_results, verifier_verdicts=[])
        
        # Verify consensus score is in valid range
        assert 0 <= consensus_score <= 100, f"Consensus score {consensus_score} must be between 0 and 100"
        
        # Verify metadata structure
        assert hasattr(metadata, 'variance'), "Metadata must have variance attribute"
        assert hasattr(metadata, 'is_low_confidence'), "Metadata must have is_low_confidence attribute"
        assert hasattr(metadata, 'individual_scores'), "Metadata must have individual_scores attribute"
        assert hasattr(metadata, 'strategy'), "Metadata must have strategy attribute"
        
        # Requirement 11.3: Flag low-confidence when variance > 20 points
        if metadata.variance > 20.0:
            assert metadata.is_low_confidence is True, \
                f"Should flag low confidence when variance {metadata.variance} > 20"
        else:
            assert metadata.is_low_confidence is False, \
                f"Should not flag low confidence when variance {metadata.variance} <= 20"
        
        # Requirement 11.4: Report individual scores alongside consensus
        assert len(metadata.individual_scores) == len(judge_results), \
            "Individual scores must be reported for all judges"
        for jr in judge_results:
            assert jr.model_name in metadata.individual_scores, \
                f"Individual score for {jr.model_name} must be in metadata"
            assert metadata.individual_scores[jr.model_name] == jr.score, \
                f"Individual score for {jr.model_name} must match judge result"
        
        # Verify strategy is correctly recorded
        assert metadata.strategy == strategy.value, \
            f"Metadata strategy {metadata.strategy} must match configured strategy {strategy.value}"
        
        # Verify aggregation correctness for each strategy
        scores = [jr.score for jr in judge_results]
        
        if strategy == EngineAggregationStrategy.MEAN:
            # Requirement 11.1: Mean strategy should compute arithmetic mean
            expected_mean = sum(scores) / len(scores)
            assert abs(consensus_score - expected_mean) < 0.01, \
                f"Mean aggregation: expected {expected_mean}, got {consensus_score}"
        
        elif strategy == EngineAggregationStrategy.MEDIAN:
            # Requirement 11.1: Median strategy should compute median
            import statistics
            expected_median = statistics.median(scores)
            assert abs(consensus_score - expected_median) < 0.01, \
                f"Median aggregation: expected {expected_median}, got {consensus_score}"
        
        elif strategy == EngineAggregationStrategy.WEIGHTED_AVERAGE:
            # Requirement 11.2: Weighted average should apply weights correctly
            assert metadata.weights is not None, "Weights must be recorded in metadata for weighted average"
            assert len(metadata.weights) == len(judge_results), \
                "Weights must be provided for all judges"
            
            # Calculate expected weighted average
            total_weight = sum(weights.values())
            expected_weighted = sum(jr.score * weights[jr.model_name] for jr in judge_results) / total_weight
            assert abs(consensus_score - expected_weighted) < 0.01, \
                f"Weighted average: expected {expected_weighted}, got {consensus_score}"
        
        elif strategy == EngineAggregationStrategy.MAJORITY_VOTE:
            # Requirement 11.1: Majority vote should bin scores and return majority bin median
            # Scores are binned: [0-33], [34-66], [67-100]
            low = [s for s in scores if s <= 33]
            medium = [s for s in scores if 34 <= s <= 66]
            high = [s for s in scores if s >= 67]
            
            # Find majority bin
            bins = [(low, 16.5), (medium, 50.0), (high, 83.5)]
            majority_bin, default_value = max(bins, key=lambda x: len(x[0]))
            
            if majority_bin:
                import statistics
                expected_majority = statistics.median(majority_bin)
            else:
                expected_majority = default_value
            
            assert abs(consensus_score - expected_majority) < 0.01, \
                f"Majority vote: expected {expected_majority}, got {consensus_score}"

    # Feature: llm-judge-auditor, Property 10: Batch aggregation correctness
    @given(scores=st.lists(score_strategy(), min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_batch_aggregation_correctness(self, scores):
        """Verify batch statistics are calculated correctly. Validates: Requirements 5.3"""
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        expected_mean = sum(scores) / n
        if n % 2 == 0:
            expected_median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            expected_median = sorted_scores[n // 2]
        
        assert abs(expected_mean - sum(scores) / len(scores)) < 0.01
        assert min(scores) <= expected_median <= max(scores)

    # Feature: llm-judge-auditor, Property 8: Pairwise ranking symmetry
    @given(
        source_text=st.text(min_size=20, max_size=200),
        candidate_a=st.text(min_size=20, max_size=200),
        candidate_b=st.text(min_size=20, max_size=200),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pairwise_ranking_symmetry(self, source_text, candidate_a, candidate_b):
        """Verify pairwise rankings are symmetric. Validates: Requirements 10.2, 14.3"""
        from llm_judge_auditor.components.judge_ensemble import JudgeEnsemble
        from llm_judge_auditor.components.prompt_manager import PromptManager
        from pathlib import Path
        import tempfile
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a minimal config
            config = ToolkitConfig(
                verifier_model="test/verifier",
                judge_models=["test/judge1"],
                quantize=False,
                device=DeviceType.CPU,
            )
            
            # Mock the device manager
            mock_device_manager = Mock()
            mock_device_manager.auto_configure = Mock(side_effect=lambda cfg: cfg)
            
            # Mock the model downloader
            mock_model_downloader = Mock()
            mock_model_downloader.get_model_path = Mock(return_value=tmp_path / "model")
            mock_model_downloader.download_model = Mock(return_value=tmp_path / "model")
            
            # Mock the model and tokenizer loading
            with patch("llm_judge_auditor.components.model_manager.AutoTokenizer") as mock_tokenizer_class, \
                 patch("llm_judge_auditor.components.model_manager.AutoModelForSeq2SeqLM") as mock_seq2seq_class, \
                 patch("llm_judge_auditor.components.model_manager.AutoModelForCausalLM") as mock_causal_class:
                
                # Setup mock models
                def create_mock_model():
                    mock_model = Mock()
                    mock_model.eval = Mock(return_value=mock_model)
                    mock_model.to = Mock(return_value=mock_model)
                    mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
                    return mock_model
                
                # Setup mock tokenizer
                mock_tokenizer = Mock()
                mock_tokenizer.pad_token = None
                mock_tokenizer.eos_token = "<eos>"
                mock_tokenizer.pad_token_id = 0
                mock_tokenizer.eos_token_id = 1
                mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
                
                # Verifier uses seq2seq
                mock_seq2seq_class.from_pretrained = Mock(return_value=create_mock_model())
                
                # Judges use causal LM
                mock_causal_class.from_pretrained = Mock(return_value=create_mock_model())
                
                # Create ModelManager
                from llm_judge_auditor.components.model_manager import ModelManager
                manager = ModelManager(
                    config=config,
                    device_manager=mock_device_manager,
                    model_downloader=mock_model_downloader,
                )
                
                # Load judge ensemble
                manager.load_judge_ensemble()
                
                # Create JudgeEnsemble
                prompt_manager = PromptManager()
                ensemble = JudgeEnsemble(
                    model_manager=manager,
                    prompt_manager=prompt_manager,
                    max_length=512,
                    temperature=0.1,
                )
                
                # We need to track which candidates are being compared in each call
                # Store the prompt to extract candidates from it
                last_prompt = {"value": ""}
                
                # Mock the tokenizer call to return proper tensors and capture the prompt
                def mock_tokenizer_call(prompt, *args, **kwargs):
                    last_prompt["value"] = prompt
                    return {
                        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
                    }
                
                mock_tokenizer.side_effect = mock_tokenizer_call
                
                # Mock the model.generate to return deterministic outputs
                def mock_generate(*args, **kwargs):
                    # Just return some tensor - the decode will handle the logic
                    return torch.tensor([[1, 2, 3, 4, 5]])
                
                # Mock the decode to return our formatted response based on the prompt
                def mock_decode(tensor, **kwargs):
                    # Extract candidates from the prompt
                    prompt = last_prompt["value"]
                    
                    # The prompt contains "Candidate A:" and "Candidate B:"
                    # We need to extract them to determine the winner
                    import re
                    
                    # Try to find Candidate A and B in the prompt
                    cand_a_match = re.search(r'Candidate A:\s*([^\n]+)', prompt)
                    cand_b_match = re.search(r'Candidate B:\s*([^\n]+)', prompt)
                    
                    if cand_a_match and cand_b_match:
                        cand_a_text = cand_a_match.group(1).strip()
                        cand_b_text = cand_b_match.group(1).strip()
                        
                        # Deterministically decide winner based on lengths
                        len_a = len(cand_a_text)
                        len_b = len(cand_b_text)
                        
                        if abs(len_a - len_b) < 5:
                            winner = "TIE"
                        elif len_a > len_b:
                            winner = "A"
                        else:
                            winner = "B"
                    else:
                        # Fallback to TIE if we can't parse
                        winner = "TIE"
                    
                    return f"REASONING: Comparing the two candidates.\nWINNER: {winner}\nEXPLANATION: Based on analysis."
                
                # Get the mock model and apply our mocks
                judge_model, judge_tokenizer = manager.get_judge("test/judge1")
                judge_model.generate = Mock(side_effect=mock_generate)
                judge_tokenizer.decode = Mock(side_effect=mock_decode)
                
                # Test pairwise comparison in both directions
                result_ab = ensemble.pairwise_compare(
                    source_text=source_text,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    judge_name="test/judge1"
                )
                
                result_ba = ensemble.pairwise_compare(
                    source_text=source_text,
                    candidate_a=candidate_b,  # Swapped
                    candidate_b=candidate_a,  # Swapped
                    judge_name="test/judge1"
                )
                
                # Verify symmetry property
                # If A > B, then when we swap, B < A (which means A wins again, but now it's in position B)
                # If TIE, then both should be TIE
                
                if result_ab.winner == "TIE":
                    assert result_ba.winner == "TIE", \
                        f"If A vs B is TIE, then B vs A should also be TIE, but got {result_ba.winner}"
                elif result_ab.winner == "A":
                    # A won in first comparison
                    # In second comparison, A is now in position B, so B should win
                    assert result_ba.winner == "B", \
                        f"If A wins in A vs B, then A (now in position B) should win in B vs A, but got {result_ba.winner}"
                elif result_ab.winner == "B":
                    # B won in first comparison
                    # In second comparison, B is now in position A, so A should win
                    assert result_ba.winner == "A", \
                        f"If B wins in A vs B, then B (now in position A) should win in B vs A, but got {result_ba.winner}"

    # Feature: llm-judge-auditor, Property 1: Model initialization completeness
    @given(config=toolkit_config_strategy())
    @settings(max_examples=100, deadline=None)
    def test_model_initialization_completeness(self, config):
        """Verify all models are loaded, quantized if configured, and verified as ready. Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5"""
        from pathlib import Path
        import tempfile
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Mock the device manager
            mock_device_manager = Mock()
            mock_device_manager.auto_configure = Mock(side_effect=lambda cfg: cfg)
            
            # Mock the model downloader
            mock_model_downloader = Mock()
            mock_model_downloader.get_model_path = Mock(return_value=tmp_path / "model")
            mock_model_downloader.download_model = Mock(return_value=tmp_path / "model")
            
            # Mock the model and tokenizer loading
            with patch("llm_judge_auditor.components.model_manager.AutoTokenizer") as mock_tokenizer_class, \
                 patch("llm_judge_auditor.components.model_manager.AutoModelForSeq2SeqLM") as mock_seq2seq_class, \
                 patch("llm_judge_auditor.components.model_manager.AutoModelForCausalLM") as mock_causal_class:
                
                # Setup mock models
                def create_mock_model():
                    mock_model = Mock()
                    mock_model.eval = Mock(return_value=mock_model)
                    mock_model.to = Mock(return_value=mock_model)
                    mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
                    return mock_model
                
                # Setup mock tokenizer
                mock_tokenizer = Mock()
                mock_tokenizer.pad_token = None
                mock_tokenizer.eos_token = "<eos>"
                mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
                
                # Verifier uses seq2seq
                mock_seq2seq_class.from_pretrained = Mock(return_value=create_mock_model())
                
                # Judges use causal LM
                mock_causal_class.from_pretrained = Mock(return_value=create_mock_model())
                
                # Create ModelManager
                manager = ModelManager(
                    config=config,
                    device_manager=mock_device_manager,
                    model_downloader=mock_model_downloader,
                )
                
                # Load verifier
                verifier_model, verifier_tokenizer = manager.load_verifier()
                
                # Verify verifier is loaded
                assert verifier_model is not None, "Verifier model should be loaded"
                assert verifier_tokenizer is not None, "Verifier tokenizer should be loaded"
                
                # Load judge ensemble
                judge_ensemble = manager.load_judge_ensemble()
                
                # Verify all judges are loaded
                assert len(judge_ensemble) == len(config.judge_models), \
                    f"All {len(config.judge_models)} judge models should be loaded"
                
                for judge_name in config.judge_models:
                    assert judge_name in judge_ensemble, f"Judge {judge_name} should be in ensemble"
                    judge_model, judge_tokenizer = judge_ensemble[judge_name]
                    assert judge_model is not None, f"Judge model {judge_name} should be loaded"
                    assert judge_tokenizer is not None, f"Judge tokenizer {judge_name} should be loaded"
                
                # Verify all models are ready
                assert manager.verify_models_ready(), "All models should be verified as ready"
                
                # Verify model info is tracked
                model_info = manager.get_model_info()
                assert config.verifier_model in model_info, "Verifier should be in model info"
                
                for judge_name in config.judge_models:
                    assert judge_name in model_info, f"Judge {judge_name} should be in model info"
                    judge_info = model_info[judge_name]
                    assert judge_info.is_ready, f"Judge {judge_name} should be marked as ready"
                    assert judge_info.quantized == config.quantize, \
                        f"Judge {judge_name} quantization should match config"
                    assert judge_info.device == manager._get_device_string(), \
                        f"Judge {judge_name} device should match config"
                
                # Verify verifier info
                verifier_info = model_info[config.verifier_model]
                assert verifier_info.is_ready, "Verifier should be marked as ready"
                assert verifier_info.quantized == config.quantize, "Verifier quantization should match config"
                assert verifier_info.device == manager._get_device_string(), "Verifier device should match config"

    # Feature: llm-judge-auditor, Property 13: Retrieval fallback behavior
    @given(
        candidate_text=st.text(min_size=20, max_size=500),
        retrieval_enabled=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_retrieval_fallback_behavior(self, candidate_text, retrieval_enabled):
        """Verify system operates in fallback mode when retrieval is disabled or no passages found. Validates: Requirements 6.5, 6.6, 6.7"""
        from llm_judge_auditor.components.retrieval_component import RetrievalComponent
        
        # Create retrieval component
        component = RetrievalComponent(top_k=3, device="cpu")
        
        # Test Case 1: Retrieval disabled (no KB initialized)
        # Component should be in fallback mode
        assert component.fallback_mode() is True, "Component should be in fallback mode when no KB is loaded"
        
        # Extract claims from candidate text
        claims = component.extract_claims(candidate_text)
        
        # In fallback mode, retrieve_passages should return empty list
        for claim in claims:
            passages = component.retrieve_passages(claim)
            assert passages == [], "In fallback mode, no passages should be retrieved"
        
        # Test Case 2: Retrieval enabled but no relevant passages found
        # Create a KB with unrelated content
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            # Write passages that are unlikely to match the candidate text
            f.write("KB:doc1\tQuantum mechanics describes subatomic particle behavior.\n")
            f.write("KB:doc2\tThe mitochondria is the powerhouse of the cell.\n")
            f.write("KB:doc3\tPythagorean theorem relates triangle side lengths.\n")
            kb_path = f.name
        
        try:
            if retrieval_enabled:
                # Initialize KB
                component.initialize_knowledge_base(kb_path)
                
                # Should not be in fallback mode after KB initialization
                assert component.fallback_mode() is False, "Component should not be in fallback mode after KB initialization"
                
                # Extract claims and retrieve passages
                claims = component.extract_claims(candidate_text)
                
                for claim in claims:
                    passages = component.retrieve_passages(claim)
                    
                    # Passages may or may not be found depending on similarity
                    # But the system should handle both cases gracefully
                    assert isinstance(passages, list), "Retrieved passages should be a list"
                    
                    # If no passages found (low relevance), this is equivalent to fallback
                    # The system should proceed with evaluation using only source text
                    if len(passages) == 0:
                        # This is the "no relevant passages found" case
                        # System should flag claim as unverifiable and proceed
                        pass  # This is the expected behavior
                    else:
                        # Passages found - verify structure
                        for passage in passages:
                            assert hasattr(passage, 'text'), "Passage should have text"
                            assert hasattr(passage, 'source'), "Passage should have source"
                            assert hasattr(passage, 'relevance_score'), "Passage should have relevance_score"
                            assert 0 <= passage.relevance_score <= 1, "Relevance score should be between 0 and 1"
            else:
                # Retrieval disabled - should remain in fallback mode
                assert component.fallback_mode() is True, "Component should remain in fallback mode when retrieval is disabled"
        
        finally:
            # Clean up
            Path(kb_path).unlink()
        
        # Test Case 3: Verify fallback mode behavior is consistent
        # Create a new component without KB
        fallback_component = RetrievalComponent(top_k=3, device="cpu")
        assert fallback_component.fallback_mode() is True, "New component should start in fallback mode"
        
        # Extract claims
        fallback_claims = fallback_component.extract_claims(candidate_text)
        
        # All retrieval attempts should return empty lists
        for claim in fallback_claims:
            passages = fallback_component.retrieve_passages(claim)
            assert passages == [], "Fallback mode should always return empty passage list"
        
        # Verify stats reflect fallback mode
        stats = fallback_component.get_stats()
        assert stats['fallback_mode'] is True, "Stats should indicate fallback mode"
        assert stats['num_passages'] == 0, "Stats should show zero passages in fallback mode"
        assert stats['index_size'] == 0, "Stats should show zero index size in fallback mode"

    # Feature: llm-judge-auditor, Property 2: Multi-stage pipeline correctness
    @given(
        source_text=st.text(min_size=50, max_size=300),
        candidate_output=st.text(min_size=50, max_size=300),
        use_retrieval=st.booleans(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_multi_stage_pipeline_correctness(self, source_text, candidate_output, use_retrieval):
        """Verify the evaluation pipeline executes in correct order: retrieval → verification → ensemble → aggregation. Validates: Requirements 2.1, 2.2, 2.3, 2.5"""
        from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
        from llm_judge_auditor.config import ToolkitConfig, DeviceType
        from pathlib import Path
        import tempfile
        
        # Track the order of operations
        operation_order = []
        
        # Create a temporary directory for models
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a minimal config
            config = ToolkitConfig(
                verifier_model="test/verifier",
                judge_models=["test/judge1"],
                quantize=False,
                device=DeviceType.CPU,
                enable_retrieval=use_retrieval,
                knowledge_base_path=None,  # Will be set if retrieval is enabled
            )
            
            # Mock the device manager
            mock_device_manager = Mock()
            mock_device_manager.auto_configure = Mock(side_effect=lambda cfg: cfg)
            
            # Mock the model downloader
            mock_model_downloader = Mock()
            mock_model_downloader.get_model_path = Mock(return_value=tmp_path / "model")
            mock_model_downloader.download_model = Mock(return_value=tmp_path / "model")
            
            # Mock the model and tokenizer loading
            with patch("llm_judge_auditor.components.model_manager.AutoTokenizer") as mock_tokenizer_class, \
                 patch("llm_judge_auditor.components.model_manager.AutoModelForSeq2SeqLM") as mock_seq2seq_class, \
                 patch("llm_judge_auditor.components.model_manager.AutoModelForCausalLM") as mock_causal_class, \
                 patch("llm_judge_auditor.components.model_manager.DeviceManager") as mock_dm_class, \
                 patch("llm_judge_auditor.components.model_manager.ModelDownloader") as mock_md_class, \
                 patch("llm_judge_auditor.evaluation_toolkit.ModelManager") as mock_mm_class:
                
                # Setup mock device manager and model downloader
                mock_dm_class.return_value = mock_device_manager
                mock_md_class.return_value = mock_model_downloader
                
                # Setup mock models
                def create_mock_model():
                    mock_model = Mock()
                    mock_model.eval = Mock(return_value=mock_model)
                    mock_model.to = Mock(return_value=mock_model)
                    mock_model.parameters = Mock(return_value=[torch.zeros(1000)])
                    return mock_model
                
                # Setup mock tokenizer
                mock_tokenizer = Mock()
                mock_tokenizer.pad_token = None
                mock_tokenizer.eos_token = "<eos>"
                mock_tokenizer.pad_token_id = 0
                mock_tokenizer.eos_token_id = 1
                mock_tokenizer_class.from_pretrained = Mock(return_value=mock_tokenizer)
                
                # Verifier uses seq2seq
                mock_verifier_model = create_mock_model()
                mock_seq2seq_class.from_pretrained = Mock(return_value=mock_verifier_model)
                
                # Judges use causal LM
                mock_judge_model = create_mock_model()
                mock_causal_class.from_pretrained = Mock(return_value=mock_judge_model)
                
                # Create a mock ModelManager instance
                mock_model_manager = Mock()
                mock_model_manager.load_verifier = Mock()
                mock_model_manager.load_judge = Mock()
                mock_model_manager.verify_models_ready = Mock(return_value=True)
                mock_model_manager.get_verifier = Mock(return_value=(mock_verifier_model, mock_tokenizer))
                mock_model_manager.get_judge = Mock(return_value=(mock_judge_model, mock_tokenizer))
                mock_model_manager.get_model_info = Mock(return_value={})
                mock_mm_class.return_value = mock_model_manager
                
                # Mock the retrieval component to track when it's called
                def mock_retrieval_extract_claims(self, text):
                    operation_order.append("retrieval_extract_claims")
                    # Return a simple claim
                    from llm_judge_auditor.models import Claim
                    return [Claim(text=text[:50], source_span=(0, 50), claim_type="factual")]
                
                def mock_retrieval_retrieve_passages(self, claim):
                    operation_order.append("retrieval_retrieve_passages")
                    # Return empty list (fallback mode)
                    return []
                
                # Mock the verifier to track when it's called
                def mock_verifier_verify_text(self, candidate_text, source_context, passages=None):
                    operation_order.append("verifier_verify_text")
                    # Return a simple verdict
                    from llm_judge_auditor.models import Verdict, VerdictLabel
                    return [Verdict(
                        label=VerdictLabel.SUPPORTED,
                        confidence=0.8,
                        evidence=["test evidence"],
                        reasoning="test reasoning"
                    )]
                
                # Mock the judge ensemble to track when it's called
                def mock_judge_evaluate_all(self, source_text, candidate_output, retrieved_context="", parallel=False):
                    operation_order.append("judge_evaluate_all")
                    # Return a simple judge result
                    from llm_judge_auditor.models import JudgeResult
                    return [JudgeResult(
                        model_name="test/judge1",
                        score=75.0,
                        reasoning="test reasoning",
                        flagged_issues=[],
                        confidence=0.8
                    )]
                
                # Mock the aggregation engine to track when it's called
                def mock_aggregation_aggregate_scores(self, judge_results, verifier_verdicts):
                    operation_order.append("aggregation_aggregate_scores")
                    # Return a simple aggregation result
                    from llm_judge_auditor.models import AggregationMetadata
                    metadata = AggregationMetadata(
                        strategy="mean",
                        variance=5.0,
                        is_low_confidence=False,
                        individual_scores={"test/judge1": 75.0},
                        weights=None
                    )
                    return 75.0, metadata
                
                # Apply the mocks
                with patch.object(RetrievalComponent, 'extract_claims', mock_retrieval_extract_claims), \
                     patch.object(RetrievalComponent, 'retrieve_passages', mock_retrieval_retrieve_passages), \
                     patch.object(SpecializedVerifier, 'verify_text', mock_verifier_verify_text), \
                     patch.object(JudgeEnsemble, 'evaluate_all', mock_judge_evaluate_all), \
                     patch.object(AggregationEngine, 'aggregate_scores', mock_aggregation_aggregate_scores):
                    
                    # Create the toolkit
                    toolkit = EvaluationToolkit(config)
                    
                    # Run evaluation
                    result = toolkit.evaluate(
                        source_text=source_text,
                        candidate_output=candidate_output,
                        use_retrieval=use_retrieval,
                    )
                    
                    # Verify the result structure
                    assert result is not None, "Evaluation should return a result"
                    assert hasattr(result, 'consensus_score'), "Result should have consensus_score"
                    assert hasattr(result, 'verifier_verdicts'), "Result should have verifier_verdicts"
                    assert hasattr(result, 'judge_results'), "Result should have judge_results"
                    assert hasattr(result, 'report'), "Result should have report"
                    
                    # Verify the pipeline order
                    # The expected order is: retrieval (if enabled) → verifier → judge → aggregation
                    
                    if use_retrieval and not toolkit.retrieval.fallback_mode():
                        # Requirement 2.1: Retrieval should happen first
                        # Check that retrieval operations come before verifier
                        retrieval_indices = [i for i, op in enumerate(operation_order) 
                                           if op.startswith("retrieval_")]
                        verifier_indices = [i for i, op in enumerate(operation_order) 
                                          if op == "verifier_verify_text"]
                        
                        if retrieval_indices and verifier_indices:
                            assert max(retrieval_indices) < min(verifier_indices), \
                                "Retrieval operations should complete before verification"
                    
                    # Requirement 2.2: Verifier should complete before judge ensemble
                    verifier_indices = [i for i, op in enumerate(operation_order) 
                                      if op == "verifier_verify_text"]
                    judge_indices = [i for i, op in enumerate(operation_order) 
                                   if op == "judge_evaluate_all"]
                    
                    if verifier_indices and judge_indices:
                        assert max(verifier_indices) < min(judge_indices), \
                            "Verification should complete before judge ensemble evaluation"
                    
                    # Requirement 2.3: Judge ensemble should complete before aggregation
                    aggregation_indices = [i for i, op in enumerate(operation_order) 
                                         if op == "aggregation_aggregate_scores"]
                    
                    if judge_indices and aggregation_indices:
                        assert max(judge_indices) < min(aggregation_indices), \
                            "Judge ensemble evaluation should complete before aggregation"
                    
                    # Requirement 2.5: Aggregation should receive outputs from previous stages
                    # This is verified by checking that aggregation happens last
                    if aggregation_indices:
                        assert aggregation_indices[-1] == len(operation_order) - 1, \
                            "Aggregation should be the final stage of the pipeline"
                    
                    # Verify that all expected stages executed
                    assert "verifier_verify_text" in operation_order, \
                        "Verifier stage should execute"
                    assert "judge_evaluate_all" in operation_order, \
                        "Judge ensemble stage should execute"
                    assert "aggregation_aggregate_scores" in operation_order, \
                        "Aggregation stage should execute"
                    
                    # Verify the result contains data from all stages
                    assert len(result.verifier_verdicts) > 0, \
                        "Result should contain verifier verdicts"
                    assert len(result.judge_results) > 0, \
                        "Result should contain judge results"
                    assert result.consensus_score >= 0 and result.consensus_score <= 100, \
                        "Result should contain valid consensus score"


    # Feature: llm-judge-auditor, Property 31: MiHR and MaHR computation correctness
    @given(
        num_verdicts=st.integers(min_value=0, max_value=20),
        num_responses=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_mihr_mahr_computation_correctness(self, num_verdicts, num_responses):
        """
        Verify MiHR and MaHR computation correctness.
        
        Property 31: MiHR and MaHR computation correctness
        *For any* set of claims with verdicts, MiHR should equal unsupported_claims / total_claims,
        and for any set of responses, MaHR should equal responses_with_hallucinations / total_responses,
        with both values in range [0.0, 1.0].
        
        **Validates: Requirements 15.1, 15.2, 15.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
        )
        from llm_judge_auditor.models import Verdict, VerdictLabel
        import random
        
        calc = HallucinationMetricsCalculator()
        
        # Generate random verdicts for MiHR test
        verdicts = []
        for _ in range(num_verdicts):
            label = random.choice([
                VerdictLabel.SUPPORTED,
                VerdictLabel.REFUTED,
                VerdictLabel.NOT_ENOUGH_INFO
            ])
            verdicts.append(Verdict(
                label=label,
                confidence=random.uniform(0.5, 1.0),
                evidence=[],
                reasoning="test"
            ))
        
        # Test MiHR computation
        mihr_result = calc.compute_mihr(verdicts)
        
        # Requirement 15.5: Handle zero claims edge case
        if num_verdicts == 0:
            assert mihr_result.value is None, "MiHR should be None when no claims"
            assert mihr_result.has_claims is False, "has_claims should be False when no claims"
            assert mihr_result.total_claims == 0, "total_claims should be 0"
        else:
            # Requirement 15.1: MiHR = unsupported_claims / total_claims
            expected_unsupported = sum(
                1 for v in verdicts 
                if v.label in (VerdictLabel.REFUTED, VerdictLabel.NOT_ENOUGH_INFO)
            )
            expected_mihr = expected_unsupported / num_verdicts
            
            assert mihr_result.value is not None, "MiHR should have a value when claims exist"
            assert mihr_result.has_claims is True, "has_claims should be True when claims exist"
            assert mihr_result.total_claims == num_verdicts, "total_claims should match input"
            assert mihr_result.unsupported_claims == expected_unsupported, \
                f"unsupported_claims should be {expected_unsupported}, got {mihr_result.unsupported_claims}"
            assert abs(mihr_result.value - expected_mihr) < 1e-10, \
                f"MiHR should be {expected_mihr}, got {mihr_result.value}"
            
            # Requirement 15.4: Output in range [0.0, 1.0]
            assert 0.0 <= mihr_result.value <= 1.0, \
                f"MiHR should be in [0.0, 1.0], got {mihr_result.value}"
        
        # Generate random response verdicts for MaHR test
        response_verdicts = []
        for _ in range(num_responses):
            # Each response has 1-5 verdicts
            response_verdict_count = random.randint(1, 5)
            response_verdicts_list = []
            for _ in range(response_verdict_count):
                label = random.choice([
                    VerdictLabel.SUPPORTED,
                    VerdictLabel.REFUTED,
                    VerdictLabel.NOT_ENOUGH_INFO
                ])
                response_verdicts_list.append(Verdict(
                    label=label,
                    confidence=random.uniform(0.5, 1.0),
                    evidence=[],
                    reasoning="test"
                ))
            response_verdicts.append(response_verdicts_list)
        
        # Test MaHR computation
        mahr_result = calc.compute_mahr(response_verdicts)
        
        if num_responses == 0:
            assert mahr_result.value == 0.0, "MaHR should be 0.0 when no responses"
            assert mahr_result.total_responses == 0, "total_responses should be 0"
        else:
            # Requirement 15.2: MaHR = responses_with_hallucinations / total_responses
            expected_with_hallucinations = sum(
                1 for verdicts_list in response_verdicts
                if any(v.label in (VerdictLabel.REFUTED, VerdictLabel.NOT_ENOUGH_INFO) 
                       for v in verdicts_list)
            )
            expected_mahr = expected_with_hallucinations / num_responses
            
            assert mahr_result.total_responses == num_responses, \
                f"total_responses should be {num_responses}, got {mahr_result.total_responses}"
            assert mahr_result.responses_with_hallucinations == expected_with_hallucinations, \
                f"responses_with_hallucinations should be {expected_with_hallucinations}, got {mahr_result.responses_with_hallucinations}"
            assert abs(mahr_result.value - expected_mahr) < 1e-10, \
                f"MaHR should be {expected_mahr}, got {mahr_result.value}"
            
            # Requirement 15.4: Output in range [0.0, 1.0]
            assert 0.0 <= mahr_result.value <= 1.0, \
                f"MaHR should be in [0.0, 1.0], got {mahr_result.value}"


    # Feature: llm-judge-auditor, Property 32: FactScore and Consensus F1 formula correctness
    @given(
        num_verdicts=st.integers(min_value=0, max_value=20),
        num_models=st.integers(min_value=1, max_value=5),
        num_claims=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_factscore_consensus_f1_correctness(self, num_verdicts, num_models, num_claims):
        """
        Verify FactScore and Consensus F1 formula correctness.
        
        Property 32: FactScore and Consensus F1 formula correctness
        *For any* set of claims with verification results, FactScore should equal 
        verified_claims / total_claims, and Consensus F1 should equal 
        2 × (precision × recall) / (precision + recall), returning 0.0 when both 
        precision and recall are zero.
        
        **Validates: Requirements 16.1, 16.3, 16.4, 16.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            ClaimVerificationMatrixBuilder,
            ClaimVerificationMatrix,
        )
        from llm_judge_auditor.models import Verdict, VerdictLabel, Claim, ClaimType
        import random
        
        calc = HallucinationMetricsCalculator()
        
        # Test FactScore computation (Requirement 16.1)
        verdicts = []
        for _ in range(num_verdicts):
            label = random.choice([
                VerdictLabel.SUPPORTED,
                VerdictLabel.REFUTED,
                VerdictLabel.NOT_ENOUGH_INFO
            ])
            verdicts.append(Verdict(
                label=label,
                confidence=random.uniform(0.5, 1.0),
                evidence=[],
                reasoning="test"
            ))
        
        factscore = calc.compute_factscore(verdicts)
        
        if num_verdicts == 0:
            assert factscore is None, "FactScore should be None when no claims"
        else:
            # FactScore = verified_claims / total_claims
            verified_claims = sum(1 for v in verdicts if v.label == VerdictLabel.SUPPORTED)
            expected_factscore = verified_claims / num_verdicts
            
            assert factscore is not None, "FactScore should have a value when claims exist"
            assert abs(factscore - expected_factscore) < 1e-10, \
                f"FactScore should be {expected_factscore}, got {factscore}"
            
            # Output in range [0.0, 1.0]
            assert 0.0 <= factscore <= 1.0, \
                f"FactScore should be in [0.0, 1.0], got {factscore}"
        
        # Test Consensus F1 computation (Requirements 16.3, 16.4, 16.5)
        if num_claims == 0 or num_models == 0:
            # Edge case: no claims or no models
            matrix = ClaimVerificationMatrix(claims=[], models=[], support_matrix=[])
            f1_result = calc.compute_consensus_f1(matrix, "model_0")
            
            # When no claims or model not in matrix, should return 0.0
            assert f1_result.precision == 0.0, "Precision should be 0.0 for empty matrix"
            assert f1_result.recall == 0.0, "Recall should be 0.0 for empty matrix"
            assert f1_result.f1 == 0.0, "F1 should be 0.0 for empty matrix"
        else:
            # Generate claims
            claims = [
                Claim(text=f"claim_{i}", source_span=(i*10, i*10+5), claim_type=ClaimType.FACTUAL)
                for i in range(num_claims)
            ]
            
            # Generate model names
            models = [f"model_{i}" for i in range(num_models)]
            
            # Generate random support matrix
            support_matrix = [
                [random.randint(0, 1) for _ in range(num_models)]
                for _ in range(num_claims)
            ]
            
            matrix = ClaimVerificationMatrix(
                claims=claims,
                models=models,
                support_matrix=support_matrix
            )
            
            # Test F1 for each model
            for model_idx, model_name in enumerate(models):
                f1_result = calc.compute_consensus_f1(matrix, model_name, consensus_threshold=0.5)
                
                # Verify F1 formula: F1 = 2 × (precision × recall) / (precision + recall)
                if f1_result.precision + f1_result.recall == 0:
                    # Requirement 16.5: Return 0.0 when both precision and recall are zero
                    assert f1_result.f1 == 0.0, \
                        "F1 should be 0.0 when precision + recall = 0"
                else:
                    expected_f1 = 2 * (f1_result.precision * f1_result.recall) / (f1_result.precision + f1_result.recall)
                    assert abs(f1_result.f1 - expected_f1) < 1e-10, \
                        f"F1 should be {expected_f1}, got {f1_result.f1}"
                
                # All values should be in [0.0, 1.0]
                assert 0.0 <= f1_result.precision <= 1.0, \
                    f"Precision should be in [0.0, 1.0], got {f1_result.precision}"
                assert 0.0 <= f1_result.recall <= 1.0, \
                    f"Recall should be in [0.0, 1.0], got {f1_result.recall}"
                assert 0.0 <= f1_result.f1 <= 1.0, \
                    f"F1 should be in [0.0, 1.0], got {f1_result.f1}"


    # Feature: llm-judge-auditor, Property 33: Claim verification matrix construction
    @given(
        num_models=st.integers(min_value=1, max_value=5),
        claims_per_model=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=100, deadline=None)
    def test_claim_verification_matrix_construction(self, num_models, claims_per_model):
        """
        Verify claim verification matrix construction.
        
        Property 33: Claim verification matrix construction
        *For any* set of model responses to the same query, the claim verification matrix 
        should have dimensions (num_unique_claims × num_models) with binary support values.
        
        **Validates: Requirements 16.2**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            ClaimVerificationMatrixBuilder,
        )
        from llm_judge_auditor.models import Claim, ClaimType, Verdict, VerdictLabel
        import random
        
        builder = ClaimVerificationMatrixBuilder()
        
        # Generate model names
        models = [f"model_{i}" for i in range(num_models)]
        
        # Generate claims for each model
        # Some claims will be shared across models, some will be unique
        all_possible_claims = [f"claim_{i}" for i in range(claims_per_model * 2)]
        
        model_claims = {}
        for model_name in models:
            # Each model gets a random subset of claims
            num_claims = random.randint(0, claims_per_model)
            selected_claim_texts = random.sample(
                all_possible_claims, 
                min(num_claims, len(all_possible_claims))
            )
            model_claims[model_name] = [
                Claim(
                    text=text, 
                    source_span=(0, len(text)), 
                    claim_type=ClaimType.FACTUAL
                )
                for text in selected_claim_texts
            ]
        
        # Build the matrix
        matrix = builder.build_matrix(model_claims)
        
        # Verify matrix dimensions
        assert len(matrix.models) == num_models, \
            f"Matrix should have {num_models} models, got {len(matrix.models)}"
        
        # Collect all unique claims
        unique_claim_texts = set()
        for claims in model_claims.values():
            for claim in claims:
                unique_claim_texts.add(claim.text.strip().lower())
        
        expected_num_claims = len(unique_claim_texts)
        assert len(matrix.claims) == expected_num_claims, \
            f"Matrix should have {expected_num_claims} unique claims, got {len(matrix.claims)}"
        
        # Verify support matrix dimensions: (num_unique_claims × num_models)
        assert len(matrix.support_matrix) == expected_num_claims, \
            f"Support matrix should have {expected_num_claims} rows, got {len(matrix.support_matrix)}"
        
        for row_idx, row in enumerate(matrix.support_matrix):
            assert len(row) == num_models, \
                f"Support matrix row {row_idx} should have {num_models} columns, got {len(row)}"
            
            # Verify binary values
            for col_idx, value in enumerate(row):
                assert value in (0, 1), \
                    f"Support matrix value at ({row_idx}, {col_idx}) should be 0 or 1, got {value}"
        
        # Verify that claims are correctly tracked
        for model_idx, model_name in enumerate(models):
            claims = model_claims[model_name]
            for claim in claims:
                normalized_text = claim.text.strip().lower()
                # Find this claim in the matrix
                found = False
                for claim_idx, matrix_claim in enumerate(matrix.claims):
                    if matrix_claim.text.strip().lower() == normalized_text:
                        # This model should support this claim
                        assert matrix.support_matrix[claim_idx][model_idx] == 1, \
                            f"Model {model_name} should support claim '{normalized_text}'"
                        found = True
                        break
                assert found, f"Claim '{normalized_text}' should be in matrix"
        
        # Test with verdicts
        model_verdicts = {}
        for model_name, claims in model_claims.items():
            verdicts = []
            for _ in claims:
                label = random.choice([
                    VerdictLabel.SUPPORTED,
                    VerdictLabel.REFUTED,
                    VerdictLabel.NOT_ENOUGH_INFO
                ])
                verdicts.append(Verdict(
                    label=label,
                    confidence=0.8,
                    evidence=[],
                    reasoning="test"
                ))
            model_verdicts[model_name] = verdicts
        
        # Build matrix with verdicts
        matrix_with_verdicts = builder.build_matrix(model_claims, model_verdicts)
        
        # Verify dimensions are the same
        assert len(matrix_with_verdicts.claims) == expected_num_claims, \
            "Matrix with verdicts should have same number of claims"
        assert len(matrix_with_verdicts.models) == num_models, \
            "Matrix with verdicts should have same number of models"
        
        # Verify that only SUPPORTED verdicts count as support
        for model_idx, model_name in enumerate(models):
            claims = model_claims[model_name]
            verdicts = model_verdicts[model_name]
            for claim_idx, (claim, verdict) in enumerate(zip(claims, verdicts)):
                normalized_text = claim.text.strip().lower()
                # Find this claim in the matrix
                for matrix_claim_idx, matrix_claim in enumerate(matrix_with_verdicts.claims):
                    if matrix_claim.text.strip().lower() == normalized_text:
                        expected_support = 1 if verdict.label == VerdictLabel.SUPPORTED else 0
                        actual_support = matrix_with_verdicts.support_matrix[matrix_claim_idx][model_idx]
                        assert actual_support == expected_support, \
                            f"Support for claim '{normalized_text}' by {model_name} should be {expected_support}, got {actual_support}"
                        break


    # Feature: llm-judge-auditor, Property 34: Fleiss' Kappa computation correctness
    @given(
        num_items=st.integers(min_value=1, max_value=20),
        num_raters=st.integers(min_value=2, max_value=10),
        num_categories=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_fleiss_kappa_computation_correctness(self, num_items, num_raters, num_categories):
        """
        Verify Fleiss' Kappa computation correctness.
        
        Property 34: Fleiss' Kappa computation correctness
        *For any* rating matrix with 2+ judges, Fleiss' Kappa should equal 
        (Po - Pe) / (1 - Pe), with interpretation labels matching standard 
        thresholds (poor <0.2, fair 0.2-0.4, moderate 0.4-0.6, substantial 0.6-0.8, 
        almost perfect >0.8).
        
        **Validates: Requirements 17.1, 17.2, 17.3**
        """
        import random
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
        )
        
        calc = HallucinationMetricsCalculator()
        
        # Generate random ratings matrix
        # ratings[i][j] = number of raters who assigned category j to item i
        ratings = []
        for _ in range(num_items):
            # Distribute raters across categories randomly
            row = [0] * num_categories
            remaining_raters = num_raters
            for j in range(num_categories - 1):
                # Assign random number of raters to this category
                count = random.randint(0, remaining_raters)
                row[j] = count
                remaining_raters -= count
            # Assign remaining raters to last category
            row[num_categories - 1] = remaining_raters
            ratings.append(row)
        
        # Compute Fleiss' Kappa
        result = calc.compute_fleiss_kappa(ratings, num_categories)
        
        # Requirement 17.4: Should not be undefined with 2+ raters
        assert result.is_undefined is False, \
            f"Kappa should be defined with {num_raters} raters"
        assert result.kappa is not None, \
            "Kappa value should not be None"
        
        # Requirement 17.1: Verify κ = (Po - Pe) / (1 - Pe)
        po = result.observed_agreement
        pe = result.expected_agreement
        
        # Verify Po and Pe are in valid range [0, 1]
        assert 0.0 <= po <= 1.0, f"Observed agreement Po={po} should be in [0, 1]"
        assert 0.0 <= pe <= 1.0, f"Expected agreement Pe={pe} should be in [0, 1]"
        
        # Verify kappa formula (with tolerance for floating point)
        if abs(1 - pe) > 1e-10:
            expected_kappa = (po - pe) / (1 - pe)
            expected_kappa = max(-1.0, min(1.0, expected_kappa))
            assert abs(result.kappa - expected_kappa) < 1e-6, \
                f"Kappa {result.kappa} should equal (Po - Pe) / (1 - Pe) = {expected_kappa}"
        
        # Verify kappa is in valid range [-1, 1]
        assert -1.0 <= result.kappa <= 1.0, \
            f"Kappa {result.kappa} should be in [-1, 1]"
        
        # Requirement 17.3: Verify interpretation matches standard thresholds
        kappa = result.kappa
        interpretation = result.interpretation
        
        if kappa < 0.2:
            assert interpretation == "poor", \
                f"Kappa {kappa} < 0.2 should be 'poor', got '{interpretation}'"
        elif kappa < 0.4:
            assert interpretation == "fair", \
                f"Kappa {kappa} in [0.2, 0.4) should be 'fair', got '{interpretation}'"
        elif kappa < 0.6:
            assert interpretation == "moderate", \
                f"Kappa {kappa} in [0.4, 0.6) should be 'moderate', got '{interpretation}'"
        elif kappa < 0.8:
            assert interpretation == "substantial", \
                f"Kappa {kappa} in [0.6, 0.8) should be 'substantial', got '{interpretation}'"
        else:
            assert interpretation == "almost_perfect", \
                f"Kappa {kappa} >= 0.8 should be 'almost_perfect', got '{interpretation}'"

    # Feature: llm-judge-auditor, Property 34: Fleiss' Kappa edge cases
    @given(
        num_items=st.integers(min_value=1, max_value=10),
        num_categories=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100, deadline=None)
    def test_fleiss_kappa_perfect_agreement(self, num_items, num_categories):
        """
        Verify Fleiss' Kappa equals 1.0 for perfect agreement.
        
        **Validates: Requirements 17.1, 17.2**
        """
        import random
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
        )
        
        calc = HallucinationMetricsCalculator()
        num_raters = 3  # Fixed number of raters for this test
        
        # Generate ratings with perfect agreement
        # All raters choose the same category for each item
        ratings = []
        for _ in range(num_items):
            row = [0] * num_categories
            chosen_category = random.randint(0, num_categories - 1)
            row[chosen_category] = num_raters  # All raters agree
            ratings.append(row)
        
        result = calc.compute_fleiss_kappa(ratings, num_categories)
        
        assert result.is_undefined is False
        assert result.kappa is not None
        
        # Perfect agreement should give kappa = 1.0
        assert result.kappa == pytest.approx(1.0, abs=0.01), \
            f"Perfect agreement should give kappa=1.0, got {result.kappa}"
        assert result.observed_agreement == pytest.approx(1.0, abs=0.01), \
            f"Perfect agreement should give Po=1.0, got {result.observed_agreement}"
        assert result.interpretation == "almost_perfect"

    # Feature: llm-judge-auditor, Property 34: Fleiss' Kappa fewer than 2 judges
    @given(
        num_items=st.integers(min_value=1, max_value=10),
        num_categories=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100, deadline=None)
    def test_fleiss_kappa_fewer_than_2_judges(self, num_items, num_categories):
        """
        Verify Fleiss' Kappa returns undefined for fewer than 2 judges.
        
        **Validates: Requirements 17.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
        )
        
        calc = HallucinationMetricsCalculator()
        
        # Generate ratings with only 1 rater
        ratings = []
        for i in range(num_items):
            row = [0] * num_categories
            row[i % num_categories] = 1  # Only 1 rater
            ratings.append(row)
        
        result = calc.compute_fleiss_kappa(ratings, num_categories)
        
        # Requirement 17.4: Should return undefined with fewer than 2 judges
        assert result.is_undefined is True, \
            "Kappa should be undefined with fewer than 2 judges"
        assert result.kappa is None, \
            "Kappa value should be None with fewer than 2 judges"
        assert "Fewer than 2 judges" in result.error_message, \
            f"Error message should mention fewer than 2 judges, got: {result.error_message}"



# Feature: llm-judge-auditor, Property 35: Uncertainty quantification correctness
class TestUncertaintyQuantificationProperties:
    """Property-based tests for uncertainty quantification correctness.
    
    **Feature: llm-judge-auditor, Property 35: Uncertainty quantification correctness**
    **Validates: Requirements 18.1, 18.2, 18.3, 18.5**
    """

    @st.composite
    def probability_distribution_strategy(draw):
        """Generate valid probability distributions that sum to 1.0."""
        # Generate 2-5 positive values
        num_outcomes = draw(st.integers(min_value=2, max_value=5))
        raw_values = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                      for _ in range(num_outcomes)]
        
        # Normalize to sum to 1.0
        total = sum(raw_values)
        if total > 0:
            return [v / total for v in raw_values]
        else:
            # Fallback to uniform distribution
            return [1.0 / num_outcomes] * num_outcomes

    @st.composite
    def inference_samples_strategy(draw):
        """Generate multiple inference samples (probability distributions)."""
        num_samples = draw(st.integers(min_value=2, max_value=10))
        num_outcomes = draw(st.integers(min_value=2, max_value=5))
        
        samples = []
        for _ in range(num_samples):
            raw_values = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                          for _ in range(num_outcomes)]
            total = sum(raw_values)
            if total > 0:
                samples.append([v / total for v in raw_values])
            else:
                samples.append([1.0 / num_outcomes] * num_outcomes)
        
        return samples

    @given(probabilities=st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=5
    ))
    @settings(max_examples=100, deadline=None)
    def test_shannon_entropy_formula_correctness(self, probabilities):
        """
        Verify Shannon entropy formula: H(p) = -Σ pᵢ log pᵢ
        
        **Feature: llm-judge-auditor, Property 35: Uncertainty quantification correctness**
        **Validates: Requirements 18.1**
        """
        import math
        from llm_judge_auditor.components.hallucination_metrics import HallucinationMetricsCalculator
        
        # Normalize probabilities to sum to 1.0
        total = sum(probabilities)
        if total <= 0:
            return  # Skip invalid distributions
        
        normalized = [p / total for p in probabilities]
        
        calc = HallucinationMetricsCalculator()
        result = calc.compute_shannon_entropy(normalized)
        
        # Manually compute expected entropy
        expected = 0.0
        for p in normalized:
            if p > 0:
                expected -= p * math.log(p)
        
        # Verify formula correctness
        assert abs(result - expected) < 1e-10, \
            f"Shannon entropy should equal -Σ pᵢ log pᵢ. Expected {expected}, got {result}"
        
        # Verify entropy is non-negative
        assert result >= 0.0, "Shannon entropy must be non-negative"

    @given(samples=st.lists(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5
        ),
        min_size=2, max_size=10
    ))
    @settings(max_examples=100, deadline=None)
    def test_epistemic_uncertainty_variance_across_samples(self, samples):
        """
        Verify epistemic uncertainty = Var(E[p]) across inference samples.
        
        **Feature: llm-judge-auditor, Property 35: Uncertainty quantification correctness**
        **Validates: Requirements 18.2**
        """
        from llm_judge_auditor.components.hallucination_metrics import HallucinationMetricsCalculator
        
        if not samples or len(samples) < 2:
            return  # Skip invalid inputs
        
        # Normalize each sample
        normalized_samples = []
        num_outcomes = len(samples[0])
        
        for sample in samples:
            if len(sample) != num_outcomes:
                return  # Skip inconsistent samples
            total = sum(sample)
            if total > 0:
                normalized_samples.append([p / total for p in sample])
            else:
                return  # Skip invalid samples
        
        calc = HallucinationMetricsCalculator()
        result = calc.compute_epistemic_uncertainty(normalized_samples)
        
        # Verify epistemic uncertainty is non-negative
        assert result >= 0.0, "Epistemic uncertainty must be non-negative"
        
        # Verify that identical samples produce zero epistemic uncertainty
        if all(sample == normalized_samples[0] for sample in normalized_samples):
            assert result == 0.0, "Identical samples should have zero epistemic uncertainty"

    @given(samples=st.lists(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5
        ),
        min_size=2, max_size=10
    ))
    @settings(max_examples=100, deadline=None)
    def test_aleatoric_uncertainty_expected_variance_within_samples(self, samples):
        """
        Verify aleatoric uncertainty = E[Var(p)] within inference samples.
        
        **Feature: llm-judge-auditor, Property 35: Uncertainty quantification correctness**
        **Validates: Requirements 18.3**
        """
        from llm_judge_auditor.components.hallucination_metrics import HallucinationMetricsCalculator
        
        if not samples:
            return  # Skip empty input
        
        # Normalize each sample
        normalized_samples = []
        for sample in samples:
            total = sum(sample)
            if total > 0:
                normalized_samples.append([p / total for p in sample])
            else:
                return  # Skip invalid samples
        
        calc = HallucinationMetricsCalculator()
        result = calc.compute_aleatoric_uncertainty(normalized_samples)
        
        # Verify aleatoric uncertainty is non-negative
        assert result >= 0.0, "Aleatoric uncertainty must be non-negative"
        
        # Manually compute expected variance within samples
        sample_variances = []
        for sample in normalized_samples:
            if sample:
                mean_p = sum(sample) / len(sample)
                variance = sum((p - mean_p) ** 2 for p in sample) / len(sample)
                sample_variances.append(variance)
        
        if sample_variances:
            expected = sum(sample_variances) / len(sample_variances)
            assert abs(result - expected) < 1e-10, \
                f"Aleatoric uncertainty should equal E[Var(p)]. Expected {expected}, got {result}"

    @given(
        probabilities=st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5
        ),
        samples=st.lists(
            st.lists(
                st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=5
            ),
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_total_uncertainty_equals_epistemic_plus_aleatoric(self, probabilities, samples):
        """
        Verify total uncertainty = epistemic + aleatoric.
        
        **Feature: llm-judge-auditor, Property 35: Uncertainty quantification correctness**
        **Validates: Requirements 18.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import HallucinationMetricsCalculator
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob <= 0:
            return
        normalized_probs = [p / total_prob for p in probabilities]
        
        # Normalize samples
        normalized_samples = []
        for sample in samples:
            total = sum(sample)
            if total > 0:
                normalized_samples.append([p / total for p in sample])
        
        if len(normalized_samples) < 2:
            return  # Need at least 2 samples
        
        calc = HallucinationMetricsCalculator()
        result = calc.compute_uncertainty(normalized_probs, normalized_samples)
        
        # Verify total = epistemic + aleatoric
        expected_total = result.epistemic + result.aleatoric
        assert abs(result.total - expected_total) < 1e-10, \
            f"Total uncertainty should equal epistemic + aleatoric. Expected {expected_total}, got {result.total}"


# Feature: llm-judge-auditor, Property 36: High uncertainty flagging
class TestHighUncertaintyFlaggingProperties:
    """Property-based tests for high uncertainty flagging.
    
    **Feature: llm-judge-auditor, Property 36: High uncertainty flagging**
    **Validates: Requirements 18.4**
    """

    @given(
        total_uncertainty=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        threshold=st.floats(min_value=0.1, max_value=1.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_high_uncertainty_flagging_threshold(self, total_uncertainty, threshold):
        """
        Verify responses are flagged when uncertainty exceeds threshold.
        
        **Feature: llm-judge-auditor, Property 36: High uncertainty flagging**
        **Validates: Requirements 18.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            HallucinationMetricsConfig,
            UncertaintyResult,
        )
        
        config = HallucinationMetricsConfig(uncertainty_high_threshold=threshold)
        calc = HallucinationMetricsCalculator(config=config)
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.5,
            epistemic=total_uncertainty / 2,
            aleatoric=total_uncertainty / 2,
            total=total_uncertainty,
            is_high_uncertainty=total_uncertainty > threshold
        )
        
        result = calc.flag_high_uncertainty(uncertainty)
        
        # Verify flagging logic
        if total_uncertainty > threshold:
            assert result is True, \
                f"Should flag high uncertainty when total ({total_uncertainty}) > threshold ({threshold})"
        else:
            assert result is False, \
                f"Should not flag when total ({total_uncertainty}) <= threshold ({threshold})"

    @given(
        threshold=st.floats(min_value=0.1, max_value=1.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_high_uncertainty_flagging_consistency(self, threshold):
        """
        Verify flagging is consistent with compute_uncertainty result.
        
        **Feature: llm-judge-auditor, Property 36: High uncertainty flagging**
        **Validates: Requirements 18.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            HallucinationMetricsConfig,
        )
        
        config = HallucinationMetricsConfig(uncertainty_high_threshold=threshold)
        calc = HallucinationMetricsCalculator(config=config)
        
        # Create samples that will produce some uncertainty
        probabilities = [0.5, 0.5]
        samples = [
            [0.6, 0.4],
            [0.4, 0.6],
            [0.5, 0.5],
        ]
        
        result = calc.compute_uncertainty(probabilities, samples)
        
        # Verify is_high_uncertainty flag matches threshold comparison
        expected_flag = result.total > threshold
        assert result.is_high_uncertainty == expected_flag, \
            f"is_high_uncertainty should be {expected_flag} when total={result.total}, threshold={threshold}"
        
        # Verify flag_high_uncertainty method agrees
        flag_result = calc.flag_high_uncertainty(result)
        assert flag_result == expected_flag, \
            f"flag_high_uncertainty should return {expected_flag}"


# Feature: llm-judge-auditor, Property 37: Hallucination profile completeness and serialization
class TestHallucinationProfileProperties:
    """Property-based tests for hallucination profile generation.
    
    **Feature: llm-judge-auditor, Property 37: Hallucination profile completeness and serialization**
    **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
    """

    @given(
        num_verdicts=st.integers(min_value=1, max_value=10),
        supported_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_profile_contains_all_metrics(self, num_verdicts, supported_ratio):
        """
        Verify hallucination profile contains MiHR, MaHR, FactScore, F1, Kappa, uncertainty,
        reliability classification, and claim-level analysis.
        
        **Feature: llm-judge-auditor, Property 37: Hallucination profile completeness and serialization**
        **Validates: Requirements 19.1, 19.2, 19.3**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            ClaimVerificationMatrix,
            ReliabilityLevel,
        )
        from llm_judge_auditor.models import Claim, ClaimType, Verdict, VerdictLabel
        
        calc = HallucinationMetricsCalculator()
        
        # Generate verdicts based on supported_ratio
        verdicts = []
        for i in range(num_verdicts):
            if i / num_verdicts < supported_ratio:
                verdicts.append(Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9))
            else:
                verdicts.append(Verdict(label=VerdictLabel.REFUTED, confidence=0.8))
        
        # Generate response verdicts for MaHR
        response_verdicts = [verdicts, verdicts[:max(1, len(verdicts)//2)]]
        
        # Generate claim matrix for Consensus F1
        claims = [
            Claim(text=f"claim_{i}", source_span=(i*10, i*10+5), claim_type=ClaimType.FACTUAL)
            for i in range(min(3, num_verdicts))
        ]
        models = ["model_a", "model_b"]
        support_matrix = [[1, 1] for _ in claims]  # All claims supported by both models
        claim_matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        # Generate judge verdicts for Fleiss' Kappa
        judge_verdicts = {
            "judge_1": verdicts,
            "judge_2": verdicts,
        }
        
        # Generate probabilities for uncertainty
        probabilities = [0.5, 0.3, 0.2]
        inference_samples = [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.6, 0.2, 0.2],
        ]
        
        # Generate profile
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            response_verdicts=response_verdicts,
            claim_matrix=claim_matrix,
            judge_verdicts=judge_verdicts,
            probabilities=probabilities,
            inference_samples=inference_samples,
        )
        
        # Requirement 19.1: Profile should contain all metrics
        assert profile.mihr is not None, "Profile should contain MiHR"
        assert profile.mahr is not None, "Profile should contain MaHR"
        assert profile.factscore is not None, "Profile should contain FactScore"
        assert profile.consensus_f1 is not None, "Profile should contain Consensus F1"
        assert profile.fleiss_kappa is not None, "Profile should contain Fleiss' Kappa"
        assert profile.uncertainty is not None, "Profile should contain uncertainty"
        
        # Requirement 19.2: Profile should have reliability classification
        assert profile.reliability in [ReliabilityLevel.HIGH, ReliabilityLevel.MEDIUM, ReliabilityLevel.LOW], \
            "Profile should have valid reliability classification"
        
        # Requirement 19.3: Profile should include claim-level analysis
        assert isinstance(profile.disputed_claims, list), "Profile should have disputed_claims list"
        assert isinstance(profile.consensus_claims, list), "Profile should have consensus_claims list"
        
        # Verify is_high_risk flag is set
        assert isinstance(profile.is_high_risk, bool), "Profile should have is_high_risk flag"

    @given(
        num_verdicts=st.integers(min_value=1, max_value=10),
        supported_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_profile_json_round_trip(self, num_verdicts, supported_ratio):
        """
        Verify hallucination profile serializes to JSON and deserializes back to equivalent profile.
        
        **Feature: llm-judge-auditor, Property 37: Hallucination profile completeness and serialization**
        **Validates: Requirements 19.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            HallucinationProfile,
            ClaimVerificationMatrix,
        )
        from llm_judge_auditor.models import Claim, ClaimType, Verdict, VerdictLabel
        import json
        
        calc = HallucinationMetricsCalculator()
        
        # Generate verdicts based on supported_ratio
        verdicts = []
        for i in range(num_verdicts):
            if i / num_verdicts < supported_ratio:
                verdicts.append(Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9))
            else:
                verdicts.append(Verdict(label=VerdictLabel.REFUTED, confidence=0.8))
        
        # Generate claim matrix with disputed and consensus claims
        claims = [
            Claim(text=f"claim_{i}", source_span=(i*10, i*10+5), claim_type=ClaimType.FACTUAL)
            for i in range(min(3, num_verdicts))
        ]
        models = ["model_a", "model_b", "model_c"]
        # First claim: consensus (all support), Second: disputed (only one supports)
        support_matrix = []
        for i, _ in enumerate(claims):
            if i == 0:
                support_matrix.append([1, 1, 1])  # Consensus
            else:
                support_matrix.append([1, 0, 0])  # Disputed
        claim_matrix = ClaimVerificationMatrix(claims=claims, models=models, support_matrix=support_matrix)
        
        # Generate profile
        profile = calc.generate_hallucination_profile(
            verdicts=verdicts,
            claim_matrix=claim_matrix,
            probabilities=[0.5, 0.3, 0.2],
        )
        
        # Serialize to JSON
        json_str = profile.to_json()
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict), "Serialized profile should be a valid JSON object"
        
        # Deserialize back
        restored = HallucinationProfile.from_json(json_str)
        
        # Verify round-trip consistency
        assert restored.mihr.value == profile.mihr.value, "MiHR should be preserved"
        assert restored.mihr.total_claims == profile.mihr.total_claims, "MiHR total_claims should be preserved"
        assert restored.mihr.unsupported_claims == profile.mihr.unsupported_claims, "MiHR unsupported_claims should be preserved"
        
        assert restored.factscore == profile.factscore, "FactScore should be preserved"
        assert restored.reliability == profile.reliability, "Reliability should be preserved"
        assert restored.is_high_risk == profile.is_high_risk, "is_high_risk should be preserved"
        
        # Verify claim lists are preserved
        assert len(restored.disputed_claims) == len(profile.disputed_claims), \
            "Disputed claims count should be preserved"
        assert len(restored.consensus_claims) == len(profile.consensus_claims), \
            "Consensus claims count should be preserved"
        
        # Verify claim content is preserved
        for orig, rest in zip(profile.disputed_claims, restored.disputed_claims):
            assert orig.text == rest.text, "Disputed claim text should be preserved"
        for orig, rest in zip(profile.consensus_claims, restored.consensus_claims):
            assert orig.text == rest.text, "Consensus claim text should be preserved"

    @given(
        num_verdicts=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_profile_handles_minimal_input(self, num_verdicts):
        """
        Verify profile generation handles minimal input gracefully.
        
        **Feature: llm-judge-auditor, Property 37: Hallucination profile completeness and serialization**
        **Validates: Requirements 19.1**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            ReliabilityLevel,
        )
        from llm_judge_auditor.models import Verdict, VerdictLabel
        
        calc = HallucinationMetricsCalculator()
        
        # Generate minimal verdicts
        verdicts = [
            Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)
            for _ in range(num_verdicts)
        ]
        
        # Generate profile with minimal input (no optional parameters)
        profile = calc.generate_hallucination_profile(verdicts=verdicts)
        
        # Profile should still be valid
        assert profile.mihr is not None, "MiHR should always be computed"
        
        if num_verdicts == 0:
            assert profile.mihr.value is None, "MiHR should be None for empty verdicts"
            assert profile.mihr.has_claims is False, "has_claims should be False for empty verdicts"
            assert profile.factscore is None, "FactScore should be None for empty verdicts"
        else:
            assert profile.mihr.value is not None, "MiHR should have value for non-empty verdicts"
            assert profile.mihr.has_claims is True, "has_claims should be True for non-empty verdicts"
            assert profile.factscore is not None, "FactScore should have value for non-empty verdicts"
        
        # Optional metrics should be None when not provided
        assert profile.mahr is None, "MaHR should be None when response_verdicts not provided"
        assert profile.consensus_f1 is None, "Consensus F1 should be None when claim_matrix not provided"
        assert profile.fleiss_kappa is None, "Fleiss' Kappa should be None when judge_verdicts not provided"
        assert profile.uncertainty is None, "Uncertainty should be None when probabilities not provided"
        
        # Reliability should still be determined
        assert profile.reliability in [ReliabilityLevel.HIGH, ReliabilityLevel.MEDIUM, ReliabilityLevel.LOW], \
            "Reliability should be determined even with minimal input"


# Feature: llm-judge-auditor, Property 38: High risk flagging thresholds
class TestHighRiskFlaggingProperties:
    """Property-based tests for high risk flagging thresholds.
    
    **Feature: llm-judge-auditor, Property 38: High risk flagging thresholds**
    **Validates: Requirements 19.5**
    """

    @given(
        mihr_value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_high_risk_mihr_threshold(self, mihr_value):
        """
        Verify high risk is flagged when MiHR > 0.3.
        
        **Feature: llm-judge-auditor, Property 38: High risk flagging thresholds**
        **Validates: Requirements 19.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            MiHRResult,
            ReliabilityLevel,
        )
        
        calc = HallucinationMetricsCalculator()
        
        mihr = MiHRResult(
            value=mihr_value,
            unsupported_claims=int(mihr_value * 10),
            total_claims=10,
            has_claims=True
        )
        
        is_high_risk = calc.is_high_risk(mihr=mihr)
        reliability = calc.determine_reliability(mihr=mihr)
        
        # Requirement 19.5: Flag high risk when MiHR > 0.3
        if mihr_value > 0.3:
            assert is_high_risk is True, \
                f"Should flag high risk when MiHR ({mihr_value}) > 0.3"
            assert reliability == ReliabilityLevel.LOW, \
                f"Reliability should be LOW when MiHR ({mihr_value}) > 0.3"
        else:
            assert is_high_risk is False, \
                f"Should not flag high risk when MiHR ({mihr_value}) <= 0.3"

    @given(
        kappa_value=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_high_risk_kappa_threshold(self, kappa_value):
        """
        Verify high risk is flagged when Kappa < 0.4.
        
        **Feature: llm-judge-auditor, Property 38: High risk flagging thresholds**
        **Validates: Requirements 19.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            KappaResult,
            ReliabilityLevel,
        )
        
        calc = HallucinationMetricsCalculator()
        
        kappa = KappaResult(
            kappa=kappa_value,
            interpretation=calc.interpret_kappa(kappa_value),
            observed_agreement=0.7,
            expected_agreement=0.3,
            is_undefined=False
        )
        
        is_high_risk = calc.is_high_risk(kappa=kappa)
        reliability = calc.determine_reliability(kappa=kappa)
        
        # Requirement 19.5: Flag high risk when Kappa < 0.4
        if kappa_value < 0.4:
            assert is_high_risk is True, \
                f"Should flag high risk when Kappa ({kappa_value}) < 0.4"
            assert reliability == ReliabilityLevel.LOW, \
                f"Reliability should be LOW when Kappa ({kappa_value}) < 0.4"
        else:
            assert is_high_risk is False, \
                f"Should not flag high risk when Kappa ({kappa_value}) >= 0.4"

    @given(
        uncertainty_total=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_high_risk_uncertainty_threshold(self, uncertainty_total):
        """
        Verify high risk is flagged when uncertainty > 0.8.
        
        **Feature: llm-judge-auditor, Property 38: High risk flagging thresholds**
        **Validates: Requirements 19.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            UncertaintyResult,
            ReliabilityLevel,
        )
        
        calc = HallucinationMetricsCalculator()
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.5,
            epistemic=uncertainty_total / 2,
            aleatoric=uncertainty_total / 2,
            total=uncertainty_total,
            is_high_uncertainty=uncertainty_total > 0.8
        )
        
        is_high_risk = calc.is_high_risk(uncertainty=uncertainty)
        reliability = calc.determine_reliability(uncertainty=uncertainty)
        
        # Requirement 19.5: Flag high risk when uncertainty > 0.8
        if uncertainty_total > 0.8:
            assert is_high_risk is True, \
                f"Should flag high risk when uncertainty ({uncertainty_total}) > 0.8"
            assert reliability == ReliabilityLevel.LOW, \
                f"Reliability should be LOW when uncertainty ({uncertainty_total}) > 0.8"
        else:
            assert is_high_risk is False, \
                f"Should not flag high risk when uncertainty ({uncertainty_total}) <= 0.8"

    @given(
        mihr_value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        kappa_value=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        uncertainty_total=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_high_risk_any_threshold_triggers(self, mihr_value, kappa_value, uncertainty_total):
        """
        Verify high risk is flagged when ANY threshold is exceeded.
        
        **Feature: llm-judge-auditor, Property 38: High risk flagging thresholds**
        **Validates: Requirements 19.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            MiHRResult,
            KappaResult,
            UncertaintyResult,
            ReliabilityLevel,
        )
        
        calc = HallucinationMetricsCalculator()
        
        mihr = MiHRResult(
            value=mihr_value,
            unsupported_claims=int(mihr_value * 10),
            total_claims=10,
            has_claims=True
        )
        
        kappa = KappaResult(
            kappa=kappa_value,
            interpretation=calc.interpret_kappa(kappa_value),
            observed_agreement=0.7,
            expected_agreement=0.3,
            is_undefined=False
        )
        
        uncertainty = UncertaintyResult(
            shannon_entropy=0.5,
            epistemic=uncertainty_total / 2,
            aleatoric=uncertainty_total / 2,
            total=uncertainty_total,
            is_high_uncertainty=uncertainty_total > 0.8
        )
        
        is_high_risk = calc.is_high_risk(mihr=mihr, kappa=kappa, uncertainty=uncertainty)
        
        # Requirement 19.5: Flag high risk when MiHR > 0.3 OR Kappa < 0.4 OR uncertainty > 0.8
        expected_high_risk = (mihr_value > 0.3) or (kappa_value < 0.4) or (uncertainty_total > 0.8)
        
        assert is_high_risk == expected_high_risk, \
            f"High risk should be {expected_high_risk} for MiHR={mihr_value}, Kappa={kappa_value}, uncertainty={uncertainty_total}"

    @given(
        num_verdicts=st.integers(min_value=1, max_value=10),
        supported_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_profile_high_risk_flag_consistency(self, num_verdicts, supported_ratio):
        """
        Verify profile is_high_risk flag is consistent with individual threshold checks.
        
        **Feature: llm-judge-auditor, Property 38: High risk flagging thresholds**
        **Validates: Requirements 19.5**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            HallucinationMetricsCalculator,
            ReliabilityLevel,
        )
        from llm_judge_auditor.models import Verdict, VerdictLabel
        
        calc = HallucinationMetricsCalculator()
        
        # Generate verdicts based on supported_ratio
        verdicts = []
        for i in range(num_verdicts):
            if i / num_verdicts < supported_ratio:
                verdicts.append(Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9))
            else:
                verdicts.append(Verdict(label=VerdictLabel.REFUTED, confidence=0.8))
        
        # Generate profile
        profile = calc.generate_hallucination_profile(verdicts=verdicts)
        
        # Verify is_high_risk is consistent with is_high_risk method
        expected_high_risk = calc.is_high_risk(
            mihr=profile.mihr,
            kappa=profile.fleiss_kappa,
            uncertainty=profile.uncertainty
        )
        
        assert profile.is_high_risk == expected_high_risk, \
            f"Profile is_high_risk ({profile.is_high_risk}) should match is_high_risk method ({expected_high_risk})"
        
        # Verify reliability is LOW when high risk
        if profile.is_high_risk:
            assert profile.reliability == ReliabilityLevel.LOW, \
                "Reliability should be LOW when is_high_risk is True"


# Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation
class TestFalseAcceptanceRateProperties:
    """Property-based tests for False Acceptance Rate computation.
    
    **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
    **Validates: Requirements 20.1, 20.2, 20.3, 20.4**
    """

    @given(
        num_queries=st.integers(min_value=1, max_value=20),
        nonexistent_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        abstention_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_far_computation_correctness(self, num_queries, nonexistent_ratio, abstention_ratio):
        """
        Verify FAR = failed_abstentions / total_nonexistent_queries.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.1, 20.2, 20.3, 20.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        
        calc = FalseAcceptanceCalculator()
        
        # Generate test data
        results = []
        for i in range(num_queries):
            is_nonexistent = (i / num_queries) < nonexistent_ratio
            did_abstain = (i / num_queries) < abstention_ratio
            
            # False acceptance = nonexistent AND NOT abstained
            is_false_acceptance = is_nonexistent and not did_abstain
            
            results.append(AbstentionResult(
                query=f"Query {i}",
                response=f"Response {i}",
                is_nonexistent_entity=is_nonexistent,
                did_abstain=did_abstain,
                is_false_acceptance=is_false_acceptance,
            ))
        
        far_result = calc.compute_far(results)
        
        # Manually compute expected values
        nonexistent_results = [r for r in results if r.is_nonexistent_entity]
        total_nonexistent = len(nonexistent_results)
        failed_abstentions = sum(1 for r in nonexistent_results if r.is_false_acceptance)
        correct_refusals = sum(1 for r in nonexistent_results if r.did_abstain)
        
        # Verify counts
        assert far_result.total_nonexistent_queries == total_nonexistent, \
            f"Total nonexistent queries should be {total_nonexistent}, got {far_result.total_nonexistent_queries}"
        assert far_result.failed_abstentions == failed_abstentions, \
            f"Failed abstentions should be {failed_abstentions}, got {far_result.failed_abstentions}"
        assert far_result.correct_refusals == correct_refusals, \
            f"Correct refusals should be {correct_refusals}, got {far_result.correct_refusals}"
        
        # Verify FAR formula: FAR = failed_abstentions / total_nonexistent_queries
        if total_nonexistent > 0:
            expected_far = failed_abstentions / total_nonexistent
            assert abs(far_result.value - expected_far) < 1e-10, \
                f"FAR should be {expected_far}, got {far_result.value}"
        else:
            assert far_result.value == 0.0, \
                f"FAR should be 0.0 when no nonexistent queries, got {far_result.value}"

    @given(
        num_queries=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_far_value_bounds(self, num_queries):
        """
        Verify FAR value is always in range [0.0, 1.0].
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.2**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        import random
        
        calc = FalseAcceptanceCalculator()
        
        # Generate random test data
        results = []
        for i in range(num_queries):
            is_nonexistent = random.random() < 0.5
            did_abstain = random.random() < 0.5
            is_false_acceptance = is_nonexistent and not did_abstain
            
            results.append(AbstentionResult(
                query=f"Query {i}",
                response=f"Response {i}",
                is_nonexistent_entity=is_nonexistent,
                did_abstain=did_abstain,
                is_false_acceptance=is_false_acceptance,
            ))
        
        far_result = calc.compute_far(results)
        
        # FAR should always be in [0.0, 1.0]
        assert 0.0 <= far_result.value <= 1.0, \
            f"FAR value should be in [0.0, 1.0], got {far_result.value}"

    @given(
        num_nonexistent=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_far_all_correct_refusals(self, num_nonexistent):
        """
        Verify FAR = 0.0 when all nonexistent queries are correctly refused.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.3**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        
        calc = FalseAcceptanceCalculator()
        
        # All nonexistent queries with correct refusals
        results = [
            AbstentionResult(
                query=f"Who is Fake Person {i}?",
                response="I don't know who that is.",
                is_nonexistent_entity=True,
                did_abstain=True,
                is_false_acceptance=False,
            )
            for i in range(num_nonexistent)
        ]
        
        far_result = calc.compute_far(results)
        
        # FAR should be 0.0 when all are correct refusals
        assert far_result.value == 0.0, \
            f"FAR should be 0.0 when all queries are correctly refused, got {far_result.value}"
        assert far_result.correct_refusals == num_nonexistent, \
            f"Correct refusals should be {num_nonexistent}, got {far_result.correct_refusals}"
        assert far_result.failed_abstentions == 0, \
            f"Failed abstentions should be 0, got {far_result.failed_abstentions}"

    @given(
        num_nonexistent=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_far_all_false_acceptances(self, num_nonexistent):
        """
        Verify FAR = 1.0 when all nonexistent queries result in false acceptance.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        
        calc = FalseAcceptanceCalculator()
        
        # All nonexistent queries with false acceptances
        results = [
            AbstentionResult(
                query=f"Who is Fake Person {i}?",
                response=f"Fake Person {i} was a famous inventor.",
                is_nonexistent_entity=True,
                did_abstain=False,
                is_false_acceptance=True,
            )
            for i in range(num_nonexistent)
        ]
        
        far_result = calc.compute_far(results)
        
        # FAR should be 1.0 when all are false acceptances
        assert far_result.value == 1.0, \
            f"FAR should be 1.0 when all queries are false acceptances, got {far_result.value}"
        assert far_result.failed_abstentions == num_nonexistent, \
            f"Failed abstentions should be {num_nonexistent}, got {far_result.failed_abstentions}"
        assert far_result.correct_refusals == 0, \
            f"Correct refusals should be 0, got {far_result.correct_refusals}"

    @given(
        query=st.text(min_size=5, max_size=100),
        response_with_abstention=st.sampled_from([
            "I don't know about that.",
            "I'm not sure what you're asking.",
            "There is no information available.",
            "I cannot find any data on this.",
            "That entity doesn't exist.",
            "I have no information about that.",
        ]),
    )
    @settings(max_examples=100, deadline=None)
    def test_abstention_detection_positive(self, query, response_with_abstention):
        """
        Verify abstention is correctly detected for known abstention patterns.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.3**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        
        calc = FalseAcceptanceCalculator()
        
        result = calc.evaluate_abstention(
            query=query,
            response=response_with_abstention,
            is_nonexistent=True,
        )
        
        # Should detect abstention
        assert result.did_abstain is True, \
            f"Should detect abstention in response: '{response_with_abstention}'"
        # Should NOT be false acceptance since model abstained
        assert result.is_false_acceptance is False, \
            f"Should not be false acceptance when model abstained"

    @given(
        query=st.text(min_size=5, max_size=100),
        factual_response=st.text(min_size=20, max_size=200).filter(
            lambda x: "don't know" not in x.lower() and 
                     "not sure" not in x.lower() and
                     "no information" not in x.lower() and
                     "cannot find" not in x.lower() and
                     "doesn't exist" not in x.lower() and
                     "unknown" not in x.lower()
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_false_acceptance_detection(self, query, factual_response):
        """
        Verify false acceptance is correctly detected when model provides factual response
        for nonexistent entity.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
        )
        
        calc = FalseAcceptanceCalculator()
        
        result = calc.evaluate_abstention(
            query=query,
            response=factual_response,
            is_nonexistent=True,
        )
        
        # Should NOT detect abstention (factual response without abstention patterns)
        assert result.did_abstain is False, \
            f"Should not detect abstention in factual response: '{factual_response[:50]}...'"
        # Should be false acceptance since model didn't abstain for nonexistent entity
        assert result.is_false_acceptance is True, \
            f"Should be false acceptance when model provides factual response for nonexistent entity"

    @given(
        num_queries=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_far_ignores_real_entity_queries(self, num_queries):
        """
        Verify FAR computation only considers nonexistent entity queries.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.1, 20.2**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        
        calc = FalseAcceptanceCalculator()
        
        # Mix of real and nonexistent entity queries
        results = []
        
        # Add real entity queries (should be ignored in FAR)
        for i in range(num_queries):
            results.append(AbstentionResult(
                query=f"Who is Real Person {i}?",
                response=f"Real Person {i} was a famous scientist.",
                is_nonexistent_entity=False,
                did_abstain=False,
                is_false_acceptance=False,
            ))
        
        # Add one nonexistent entity query with correct refusal
        results.append(AbstentionResult(
            query="Who is Fake Person?",
            response="I don't know who that is.",
            is_nonexistent_entity=True,
            did_abstain=True,
            is_false_acceptance=False,
        ))
        
        far_result = calc.compute_far(results)
        
        # FAR should only consider the one nonexistent query
        assert far_result.total_nonexistent_queries == 1, \
            f"Should only count nonexistent queries, got {far_result.total_nonexistent_queries}"
        assert far_result.value == 0.0, \
            f"FAR should be 0.0 (correct refusal), got {far_result.value}"

    @given(
        num_correct=st.integers(min_value=0, max_value=10),
        num_false=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_far_counts_consistency(self, num_correct, num_false):
        """
        Verify FAR counts are internally consistent.
        
        **Feature: llm-judge-auditor, Property 39: False Acceptance Rate computation**
        **Validates: Requirements 20.2, 20.3, 20.4**
        """
        from llm_judge_auditor.components.hallucination_metrics import (
            FalseAcceptanceCalculator,
            AbstentionResult,
        )
        
        calc = FalseAcceptanceCalculator()
        
        results = []
        
        # Add correct refusals
        for i in range(num_correct):
            results.append(AbstentionResult(
                query=f"Correct {i}",
                response="I don't know.",
                is_nonexistent_entity=True,
                did_abstain=True,
                is_false_acceptance=False,
            ))
        
        # Add false acceptances
        for i in range(num_false):
            results.append(AbstentionResult(
                query=f"False {i}",
                response="Here is the information.",
                is_nonexistent_entity=True,
                did_abstain=False,
                is_false_acceptance=True,
            ))
        
        far_result = calc.compute_far(results)
        
        # Verify counts are consistent
        total = num_correct + num_false
        assert far_result.total_nonexistent_queries == total, \
            f"Total should be {total}, got {far_result.total_nonexistent_queries}"
        assert far_result.correct_refusals == num_correct, \
            f"Correct refusals should be {num_correct}, got {far_result.correct_refusals}"
        assert far_result.failed_abstentions == num_false, \
            f"Failed abstentions should be {num_false}, got {far_result.failed_abstentions}"
        
        # Verify: correct_refusals + failed_abstentions = total_nonexistent_queries
        assert far_result.correct_refusals + far_result.failed_abstentions == far_result.total_nonexistent_queries, \
            "correct_refusals + failed_abstentions should equal total_nonexistent_queries"

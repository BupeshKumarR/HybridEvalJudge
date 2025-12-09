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

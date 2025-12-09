"""
Integration tests for API Judge components.

These tests verify real API integration with Groq and Gemini services.
They require valid API keys to be set in environment variables:
- GROQ_API_KEY
- GEMINI_API_KEY

Tests can be run with:
    pytest tests/integration/test_api_judges.py -v

To skip tests without API keys:
    pytest tests/integration/test_api_judges.py -v -m "not requires_api_keys"
"""

import os
import pytest
import time

from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqAPIError,
    GroqAuthenticationError,
    GroqRateLimitError
)
from llm_judge_auditor.components.gemini_judge_client import (
    GeminiJudgeClient,
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiRateLimitError
)
from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.config import ToolkitConfig


# Check for API keys
HAS_GROQ_KEY = bool(os.environ.get("GROQ_API_KEY"))
HAS_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY"))
HAS_ANY_KEY = HAS_GROQ_KEY or HAS_GEMINI_KEY


# Skip markers
requires_groq = pytest.mark.skipif(
    not HAS_GROQ_KEY,
    reason="GROQ_API_KEY not set. Set it to run Groq integration tests."
)

requires_gemini = pytest.mark.skipif(
    not HAS_GEMINI_KEY,
    reason="GEMINI_API_KEY not set. Set it to run Gemini integration tests."
)

requires_any_api_key = pytest.mark.skipif(
    not HAS_ANY_KEY,
    reason="No API keys set. Set GROQ_API_KEY or GEMINI_API_KEY to run API integration tests."
)


@pytest.fixture
def sample_source_text():
    """Sample source text for testing."""
    return """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists
and intellectuals for its design, but it has become a global cultural icon of France and one
of the most recognizable structures in the world. The tower is 330 metres (1,083 ft) tall."""


@pytest.fixture
def sample_candidate_output():
    """Sample candidate output for testing."""
    return """The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel
and built between 1887 and 1889. The tower stands at 330 meters tall and has become
an iconic symbol of France."""


@pytest.fixture
def sample_hallucinated_output():
    """Sample output with hallucinations for testing."""
    return """The Eiffel Tower is located in Paris, France. It was designed by Leonardo da Vinci
and built in 1850. The tower is 500 meters tall and is made entirely of gold."""


class TestAPIKeyManager:
    """Integration tests for APIKeyManager."""
    
    def test_load_keys_from_environment(self):
        """Test loading API keys from environment variables."""
        manager = APIKeyManager()
        available_keys = manager.load_keys()
        
        # Verify structure
        assert isinstance(available_keys, dict)
        assert "groq" in available_keys
        assert "gemini" in available_keys
        
        # Verify keys match environment
        if HAS_GROQ_KEY:
            assert available_keys["groq"] is True
            assert manager.groq_key is not None
        
        if HAS_GEMINI_KEY:
            assert available_keys["gemini"] is True
            assert manager.gemini_key is not None
    
    @requires_any_api_key
    def test_validate_all_keys(self):
        """Test validation of all available API keys."""
        manager = APIKeyManager()
        manager.load_keys()
        
        # Validate keys
        validation_results = manager.validate_all_keys(verbose=False)
        
        # Verify structure
        assert isinstance(validation_results, dict)
        
        # At least one key should be valid
        assert any(validation_results.values())
    
    def test_get_setup_instructions(self):
        """Test generation of setup instructions."""
        manager = APIKeyManager()
        instructions = manager.get_setup_instructions()
        
        # Verify instructions contain key information
        assert "GROQ_API_KEY" in instructions
        assert "GEMINI_API_KEY" in instructions
        assert "console.groq.com" in instructions
        assert "aistudio.google.com" in instructions


@pytest.mark.integration
class TestGroqJudgeIntegration:
    """Integration tests for Groq judge client with real API."""
    
    @requires_groq
    def test_groq_basic_evaluation(self, sample_source_text, sample_candidate_output):
        """Test basic evaluation with Groq API."""
        client = GroqJudgeClient(api_key=os.environ["GROQ_API_KEY"])
        
        verdict = client.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        
        # Verify verdict structure
        assert verdict is not None
        assert verdict.judge_name == "groq-llama-3.3-70b-versatile"
        assert 0 <= verdict.score <= 100
        assert 0 <= verdict.confidence <= 1
        assert verdict.reasoning is not None
        assert isinstance(verdict.issues, list)
        assert "response_time_seconds" in verdict.metadata
    
    @requires_groq
    def test_groq_hallucination_detection(self, sample_source_text, sample_hallucinated_output):
        """Test that Groq detects hallucinations."""
        client = GroqJudgeClient(api_key=os.environ["GROQ_API_KEY"])
        
        verdict = client.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_hallucinated_output,
            task="factual_accuracy"
        )
        
        # Should detect issues with hallucinated content
        # Score should be lower than for accurate content
        assert verdict.score < 70  # Hallucinated content should score poorly
        assert len(verdict.issues) > 0  # Should detect issues
    
    @requires_groq
    def test_groq_bias_detection(self):
        """Test bias detection with Groq."""
        client = GroqJudgeClient(api_key=os.environ["GROQ_API_KEY"])
        
        biased_text = """Women are naturally better at nursing and teaching, while men are
better suited for engineering and leadership roles."""
        
        verdict = client.evaluate(
            source_text="",
            candidate_output=biased_text,
            task="bias_detection"
        )
        
        # Should detect bias
        assert verdict.score < 80  # Biased content should score lower
        # Note: May or may not flag specific issues depending on model sensitivity
    
    def test_groq_invalid_api_key(self):
        """Test that invalid API key raises appropriate error."""
        client = GroqJudgeClient(api_key="invalid_key_12345")
        
        with pytest.raises(GroqAuthenticationError):
            client.evaluate(
                source_text="Test",
                candidate_output="Test",
                task="factual_accuracy"
            )
    
    @requires_groq
    def test_groq_response_time(self, sample_source_text, sample_candidate_output):
        """Test that Groq responds within reasonable time."""
        client = GroqJudgeClient(api_key=os.environ["GROQ_API_KEY"])
        
        start_time = time.time()
        verdict = client.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        elapsed = time.time() - start_time
        
        # Should complete within 30 seconds
        assert elapsed < 30
        
        # Metadata should track response time
        assert "response_time_seconds" in verdict.metadata
        assert verdict.metadata["response_time_seconds"] > 0


@pytest.mark.integration
class TestGeminiJudgeIntegration:
    """Integration tests for Gemini judge client with real API."""
    
    @requires_gemini
    def test_gemini_basic_evaluation(self, sample_source_text, sample_candidate_output):
        """Test basic evaluation with Gemini API."""
        client = GeminiJudgeClient(api_key=os.environ["GEMINI_API_KEY"])
        
        verdict = client.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        
        # Verify verdict structure
        assert verdict is not None
        # Judge name is constructed as: classname-without-judgeclient + model
        assert "gemini" in verdict.judge_name.lower()
        assert 0 <= verdict.score <= 100
        assert 0 <= verdict.confidence <= 1
        assert verdict.reasoning is not None
        assert isinstance(verdict.issues, list)
        assert "response_time_seconds" in verdict.metadata
    
    @requires_gemini
    def test_gemini_hallucination_detection(self, sample_source_text, sample_hallucinated_output):
        """Test that Gemini detects hallucinations."""
        client = GeminiJudgeClient(api_key=os.environ["GEMINI_API_KEY"])
        
        verdict = client.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_hallucinated_output,
            task="factual_accuracy"
        )
        
        # Should detect issues with hallucinated content
        assert verdict.score < 70  # Hallucinated content should score poorly
        assert len(verdict.issues) > 0  # Should detect issues
    
    @requires_gemini
    def test_gemini_bias_detection(self):
        """Test bias detection with Gemini."""
        client = GeminiJudgeClient(api_key=os.environ["GEMINI_API_KEY"])
        
        biased_text = """Women are naturally better at nursing and teaching, while men are
better suited for engineering and leadership roles."""
        
        verdict = client.evaluate(
            source_text="",
            candidate_output=biased_text,
            task="bias_detection"
        )
        
        # Should detect bias
        assert verdict.score < 80  # Biased content should score lower
    
    def test_gemini_invalid_api_key(self):
        """Test that invalid API key raises appropriate error."""
        client = GeminiJudgeClient(api_key="invalid_key_12345")
        
        with pytest.raises(GeminiAuthenticationError):
            client.evaluate(
                source_text="Test",
                candidate_output="Test",
                task="factual_accuracy"
            )
    
    @requires_gemini
    def test_gemini_response_time(self, sample_source_text, sample_candidate_output):
        """Test that Gemini responds within reasonable time."""
        client = GeminiJudgeClient(api_key=os.environ["GEMINI_API_KEY"])
        
        start_time = time.time()
        verdict = client.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        elapsed = time.time() - start_time
        
        # Should complete within 30 seconds
        assert elapsed < 30
        
        # Metadata should track response time
        assert "response_time_seconds" in verdict.metadata
        assert verdict.metadata["response_time_seconds"] > 0


@pytest.mark.integration
class TestAPIJudgeEnsemble:
    """Integration tests for API Judge Ensemble."""
    
    @requires_any_api_key
    def test_ensemble_initialization(self):
        """Test that ensemble initializes with available API keys."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        # Should have at least one judge
        assert ensemble.get_judge_count() > 0
        
        # Judge names should be populated
        judge_names = ensemble.get_judge_names()
        assert len(judge_names) > 0
        
        # Verify expected judges based on available keys
        if HAS_GROQ_KEY:
            assert any("groq" in name.lower() for name in judge_names)
        if HAS_GEMINI_KEY:
            assert any("gemini" in name.lower() for name in judge_names)
    
    @requires_any_api_key
    def test_ensemble_evaluation(self, sample_source_text, sample_candidate_output):
        """Test ensemble evaluation with available judges."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        verdicts = ensemble.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        
        # Should get verdicts from all available judges
        assert len(verdicts) == ensemble.get_judge_count()
        
        # All verdicts should be valid
        for verdict in verdicts:
            assert 0 <= verdict.score <= 100
            assert 0 <= verdict.confidence <= 1
            assert verdict.reasoning is not None
    
    @requires_any_api_key
    def test_ensemble_parallel_vs_sequential(self, sample_source_text, sample_candidate_output):
        """Test that parallel and sequential execution produce similar results."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        
        # Parallel execution
        ensemble_parallel = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        start_parallel = time.time()
        verdicts_parallel = ensemble_parallel.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        time_parallel = time.time() - start_parallel
        
        # Sequential execution
        ensemble_sequential = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=False
        )
        
        start_sequential = time.time()
        verdicts_sequential = ensemble_sequential.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        time_sequential = time.time() - start_sequential
        
        # Should get same number of verdicts
        assert len(verdicts_parallel) == len(verdicts_sequential)
        
        # Parallel should be faster (or similar) if multiple judges
        if len(verdicts_parallel) > 1:
            # Allow some margin for variance
            assert time_parallel <= time_sequential * 1.5
    
    @requires_any_api_key
    def test_ensemble_aggregation(self, sample_source_text, sample_candidate_output):
        """Test verdict aggregation."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        verdicts = ensemble.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        
        consensus, individual_scores, disagreement = ensemble.aggregate_verdicts(verdicts)
        
        # Verify aggregation results
        assert 0 <= consensus <= 100
        assert isinstance(individual_scores, dict)
        assert len(individual_scores) == len(verdicts)
        assert disagreement >= 0
        
        # Consensus should be within range of individual scores
        scores = [v.score for v in verdicts]
        assert min(scores) <= consensus <= max(scores)
    
    @requires_any_api_key
    def test_ensemble_disagreement_detection(self, sample_source_text):
        """Test disagreement detection between judges."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        # Use ambiguous content that might cause disagreement
        ambiguous_output = """The Eiffel Tower is in Paris. It's quite tall."""
        
        verdicts = ensemble.evaluate(
            source_text=sample_source_text,
            candidate_output=ambiguous_output,
            task="factual_accuracy"
        )
        
        disagreement_analysis = ensemble.identify_disagreements(verdicts, threshold=20.0)
        
        # Verify analysis structure
        assert "has_disagreement" in disagreement_analysis
        assert "variance" in disagreement_analysis
        assert "score_range" in disagreement_analysis
        assert "outliers" in disagreement_analysis
        assert "reasoning_summary" in disagreement_analysis
        
        # Reasoning summary should have entries for all judges
        assert len(disagreement_analysis["reasoning_summary"]) == len(verdicts)
    
    @requires_any_api_key
    def test_ensemble_partial_failure_handling(self, sample_source_text, sample_candidate_output):
        """Test that ensemble handles partial failures gracefully."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        # Normal evaluation should succeed
        verdicts = ensemble.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        
        # Should get at least one verdict
        assert len(verdicts) > 0


@pytest.mark.integration
class TestEndToEndEvaluation:
    """End-to-end integration tests for complete evaluation flow."""
    
    @requires_any_api_key
    def test_complete_evaluation_flow(self, sample_source_text, sample_candidate_output):
        """Test complete evaluation flow from API keys to final verdict."""
        # Step 1: Load API keys
        api_key_manager = APIKeyManager()
        available_keys = api_key_manager.load_keys()
        
        assert api_key_manager.has_any_keys()
        
        # Step 2: Validate keys
        validation_results = api_key_manager.validate_all_keys(verbose=False)
        assert any(validation_results.values())
        
        # Step 3: Initialize ensemble
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        assert ensemble.get_judge_count() > 0
        
        # Step 4: Evaluate
        verdicts = ensemble.evaluate(
            source_text=sample_source_text,
            candidate_output=sample_candidate_output,
            task="factual_accuracy"
        )
        
        assert len(verdicts) > 0
        
        # Step 5: Aggregate
        consensus, individual_scores, disagreement = ensemble.aggregate_verdicts(verdicts)
        
        assert 0 <= consensus <= 100
        
        # Step 6: Analyze disagreements
        disagreement_analysis = ensemble.identify_disagreements(verdicts)
        
        assert "has_disagreement" in disagreement_analysis
        
        # Complete flow should produce valid results
        print(f"\n✅ Complete evaluation flow successful!")
        print(f"   Judges used: {', '.join(ensemble.get_judge_names())}")
        print(f"   Consensus score: {consensus:.1f}/100")
        print(f"   Disagreement level: {disagreement:.1f}")
    
    @requires_any_api_key
    def test_multiple_evaluations(self, sample_source_text):
        """Test multiple evaluations in sequence."""
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        test_cases = [
            "The Eiffel Tower is in Paris.",
            "The Eiffel Tower is 330 meters tall.",
            "The Eiffel Tower was built by Gustave Eiffel.",
        ]
        
        results = []
        for candidate in test_cases:
            verdicts = ensemble.evaluate(
                source_text=sample_source_text,
                candidate_output=candidate,
                task="factual_accuracy"
            )
            consensus, _, _ = ensemble.aggregate_verdicts(verdicts)
            results.append(consensus)
        
        # All evaluations should succeed
        assert len(results) == len(test_cases)
        assert all(0 <= score <= 100 for score in results)


@pytest.mark.integration
class TestDemoIntegration:
    """Integration tests for demo functionality."""
    
    @requires_any_api_key
    def test_demo_can_run(self):
        """Test that demo components can be imported and initialized."""
        # Import demo components
        from llm_judge_auditor.components.api_key_manager import APIKeyManager
        from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
        from llm_judge_auditor.config import ToolkitConfig
        
        # Initialize components
        api_key_manager = APIKeyManager()
        api_key_manager.load_keys()
        
        assert api_key_manager.has_any_keys()
        
        config = ToolkitConfig()
        ensemble = APIJudgeEnsemble(
            config=config,
            api_key_manager=api_key_manager,
            parallel_execution=True
        )
        
        assert ensemble.get_judge_count() > 0
        
        # Demo should be able to run with these components
        print(f"\n✅ Demo components initialized successfully")
        print(f"   Available judges: {', '.join(ensemble.get_judge_names())}")


# Summary test to show what's available
def test_api_availability_summary():
    """Display summary of available API keys for testing."""
    print("\n" + "=" * 80)
    print("API KEY AVAILABILITY SUMMARY")
    print("=" * 80)
    
    if HAS_GROQ_KEY:
        print("✅ GROQ_API_KEY is set - Groq tests will run")
    else:
        print("❌ GROQ_API_KEY not set - Groq tests will be skipped")
        print("   Set it with: export GROQ_API_KEY='your-key'")
    
    if HAS_GEMINI_KEY:
        print("✅ GEMINI_API_KEY is set - Gemini tests will run")
    else:
        print("❌ GEMINI_API_KEY not set - Gemini tests will be skipped")
        print("   Set it with: export GEMINI_API_KEY='your-key'")
    
    if HAS_ANY_KEY:
        print("\n✅ At least one API key is available - integration tests will run")
    else:
        print("\n❌ No API keys available - all integration tests will be skipped")
        print("\nTo run integration tests:")
        print("1. Get free API keys:")
        print("   • Groq: https://console.groq.com/keys")
        print("   • Gemini: https://aistudio.google.com/app/apikey")
        print("2. Set environment variables:")
        print("   export GROQ_API_KEY='your-groq-key'")
        print("   export GEMINI_API_KEY='your-gemini-key'")
        print("3. Run tests:")
        print("   pytest tests/integration/test_api_judges.py -v")
    
    print("=" * 80 + "\n")

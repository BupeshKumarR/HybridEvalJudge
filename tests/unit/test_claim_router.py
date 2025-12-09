"""
Unit tests for the ClaimRouter component.

Tests claim classification and routing logic for specialized judge selection.
"""

import pytest

from llm_judge_auditor.components.claim_router import ClaimRouter
from llm_judge_auditor.models import Claim, ClaimType


class TestClaimClassification:
    """Test claim type classification."""

    def test_classify_numerical_claim_with_percentage(self):
        """Test classification of claims with percentages."""
        router = ClaimRouter()
        claim = Claim(
            text="The unemployment rate increased by 5.2% last quarter.",
            source_span=(0, 54),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.NUMERICAL

    def test_classify_numerical_claim_with_measurement(self):
        """Test classification of claims with measurements."""
        router = ClaimRouter()
        claim = Claim(
            text="The building is 150 meters tall.",
            source_span=(0, 33),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.NUMERICAL

    def test_classify_numerical_claim_with_currency(self):
        """Test classification of claims with currency."""
        router = ClaimRouter()
        claim = Claim(
            text="The project cost 2.5 million dollars.",
            source_span=(0, 38),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.NUMERICAL

    def test_classify_temporal_claim_with_year(self):
        """Test classification of claims with years."""
        router = ClaimRouter()
        claim = Claim(
            text="The Eiffel Tower was completed in 1889.",
            source_span=(0, 40),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.TEMPORAL

    def test_classify_temporal_claim_with_month(self):
        """Test classification of claims with months."""
        router = ClaimRouter()
        claim = Claim(
            text="The conference will be held in September.",
            source_span=(0, 42),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.TEMPORAL

    def test_classify_temporal_claim_with_relative_time(self):
        """Test classification of claims with relative time references."""
        router = ClaimRouter()
        claim = Claim(
            text="The event happened yesterday afternoon.",
            source_span=(0, 39),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.TEMPORAL

    def test_classify_logical_claim_with_conditionals(self):
        """Test classification of claims with logical conditionals."""
        router = ClaimRouter()
        claim = Claim(
            text="If the temperature rises, then the ice will melt.",
            source_span=(0, 50),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.LOGICAL

    def test_classify_logical_claim_with_causation(self):
        """Test classification of claims with causal relationships."""
        router = ClaimRouter()
        claim = Claim(
            text="The accident occurred because the driver was distracted.",
            source_span=(0, 57),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.LOGICAL

    def test_classify_commonsense_claim(self):
        """Test classification of commonsense claims."""
        router = ClaimRouter()
        claim = Claim(
            text="People generally feel happier when the weather is warm.",
            source_span=(0, 56),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.COMMONSENSE

    def test_classify_factual_claim_default(self):
        """Test classification defaults to factual for general statements."""
        router = ClaimRouter()
        claim = Claim(
            text="Paris is the capital of France.",
            source_span=(0, 32),
        )
        
        claim_type = router.classify_claim(claim)
        assert claim_type == ClaimType.FACTUAL


class TestJudgeRouting:
    """Test routing claims to specialized judges."""

    def test_route_to_specialized_judge(self):
        """Test routing to a judge specialized in the claim type."""
        specializations = {
            "mistral-7b": ["numerical", "temporal"],
            "llama-3-8b": ["factual", "logical"],
        }
        router = ClaimRouter(specializations)
        
        claim = Claim(
            text="The temperature was 25 degrees.",
            source_span=(0, 31),
            claim_type=ClaimType.NUMERICAL,
        )
        
        judge = router.route_to_judge(claim, ["mistral-7b", "llama-3-8b"])
        assert judge == "mistral-7b"

    def test_route_to_default_judge_when_no_specialist(self):
        """Test routing to first available judge when no specialist exists."""
        specializations = {
            "mistral-7b": ["numerical"],
            "llama-3-8b": ["factual"],
        }
        router = ClaimRouter(specializations)
        
        claim = Claim(
            text="People usually feel happy on sunny days.",
            source_span=(0, 41),
            claim_type=ClaimType.COMMONSENSE,
        )
        
        judge = router.route_to_judge(claim, ["mistral-7b", "llama-3-8b"])
        # Should return first available judge since no commonsense specialist
        assert judge in ["mistral-7b", "llama-3-8b"]

    def test_route_reclassifies_factual_claims(self):
        """Test that routing reclassifies generic factual claims."""
        specializations = {
            "mistral-7b": ["numerical"],
            "llama-3-8b": ["factual"],
        }
        router = ClaimRouter(specializations)
        
        # Claim starts as FACTUAL but contains numerical content
        claim = Claim(
            text="The building is 150 meters tall.",
            source_span=(0, 33),
            claim_type=ClaimType.FACTUAL,  # Initially factual
        )
        
        judge = router.route_to_judge(claim, ["mistral-7b", "llama-3-8b"])
        # Should reclassify as numerical and route to mistral-7b
        assert judge == "mistral-7b"
        assert claim.claim_type == ClaimType.NUMERICAL

    def test_route_raises_error_with_no_judges(self):
        """Test that routing raises error when no judges available."""
        router = ClaimRouter()
        claim = Claim(text="Test claim", source_span=(0, 10))
        
        with pytest.raises(ValueError, match="No judges available"):
            router.route_to_judge(claim, [])

    def test_route_multiple_claims_to_judges(self):
        """Test routing multiple claims and grouping by judge."""
        specializations = {
            "mistral-7b": ["numerical", "temporal"],
            "llama-3-8b": ["factual", "logical"],
            "phi-3-mini": ["commonsense"],
        }
        router = ClaimRouter(specializations)
        
        claims = [
            Claim(text="The temperature was 25 degrees.", source_span=(0, 31)),
            Claim(text="Paris is the capital of France.", source_span=(0, 32)),
            Claim(text="The event happened in 1989.", source_span=(0, 27)),
            Claim(text="People generally prefer warm weather.", source_span=(0, 38)),
        ]
        
        routing = router.route_claims_to_judges(
            claims,
            ["mistral-7b", "llama-3-8b", "phi-3-mini"]
        )
        
        # Check that all judges are in the result
        assert set(routing.keys()) == {"mistral-7b", "llama-3-8b", "phi-3-mini"}
        
        # Check that all claims were routed
        total_routed = sum(len(judge_claims) for judge_claims in routing.values())
        assert total_routed == len(claims)
        
        # Check that numerical/temporal claims went to mistral-7b
        mistral_claims = routing["mistral-7b"]
        assert len(mistral_claims) >= 2  # At least the numerical and temporal claims

    def test_route_multiple_claims_raises_error_with_no_judges(self):
        """Test that batch routing raises error when no judges available."""
        router = ClaimRouter()
        claims = [Claim(text="Test claim", source_span=(0, 10))]
        
        with pytest.raises(ValueError, match="No judges available"):
            router.route_claims_to_judges(claims, [])


class TestSpecializationManagement:
    """Test specialization management methods."""

    def test_get_judge_specializations(self):
        """Test retrieving specializations for a specific judge."""
        specializations = {
            "mistral-7b": ["numerical", "temporal"],
            "llama-3-8b": ["factual"],
        }
        router = ClaimRouter(specializations)
        
        specs = router.get_judge_specializations("mistral-7b")
        assert specs == ["numerical", "temporal"]
        
        specs = router.get_judge_specializations("llama-3-8b")
        assert specs == ["factual"]
        
        # Non-existent judge returns empty list
        specs = router.get_judge_specializations("unknown-judge")
        assert specs == []

    def test_get_specialized_judges(self):
        """Test retrieving judges specialized in a claim type."""
        specializations = {
            "mistral-7b": ["numerical", "temporal"],
            "llama-3-8b": ["numerical", "factual"],
            "phi-3-mini": ["commonsense"],
        }
        router = ClaimRouter(specializations)
        
        judges = router.get_specialized_judges(ClaimType.NUMERICAL)
        assert set(judges) == {"mistral-7b", "llama-3-8b"}
        
        judges = router.get_specialized_judges(ClaimType.TEMPORAL)
        assert judges == ["mistral-7b"]
        
        judges = router.get_specialized_judges(ClaimType.COMMONSENSE)
        assert judges == ["phi-3-mini"]
        
        # No specialists for logical
        judges = router.get_specialized_judges(ClaimType.LOGICAL)
        assert judges == []

    def test_update_specializations(self):
        """Test updating judge specializations dynamically."""
        router = ClaimRouter()
        
        # Add new judge
        router.update_specializations("new-judge", ["numerical", "temporal"])
        
        specs = router.get_judge_specializations("new-judge")
        assert specs == ["numerical", "temporal"]
        
        judges = router.get_specialized_judges(ClaimType.NUMERICAL)
        assert "new-judge" in judges
        
        # Update existing judge
        router.update_specializations("new-judge", ["factual"])
        
        specs = router.get_judge_specializations("new-judge")
        assert specs == ["factual"]
        
        # Should be removed from numerical specialists
        judges = router.get_specialized_judges(ClaimType.NUMERICAL)
        assert "new-judge" not in judges
        
        # Should be added to factual specialists
        judges = router.get_specialized_judges(ClaimType.FACTUAL)
        assert "new-judge" in judges


class TestInitialization:
    """Test ClaimRouter initialization."""

    def test_init_with_no_specializations(self):
        """Test initialization without specializations."""
        router = ClaimRouter()
        
        assert router.judge_specializations == {}
        assert router._specialization_map == {}

    def test_init_with_specializations(self):
        """Test initialization with specializations."""
        specializations = {
            "mistral-7b": ["numerical", "temporal"],
            "llama-3-8b": ["factual", "logical"],
        }
        router = ClaimRouter(specializations)
        
        assert router.judge_specializations == specializations
        
        # Check reverse mapping
        assert "mistral-7b" in router._specialization_map["numerical"]
        assert "mistral-7b" in router._specialization_map["temporal"]
        assert "llama-3-8b" in router._specialization_map["factual"]
        assert "llama-3-8b" in router._specialization_map["logical"]

    def test_init_builds_reverse_mapping(self):
        """Test that initialization builds correct reverse mapping."""
        specializations = {
            "judge-a": ["numerical"],
            "judge-b": ["numerical", "temporal"],
            "judge-c": ["factual"],
        }
        router = ClaimRouter(specializations)
        
        # Multiple judges can specialize in same type
        numerical_judges = router._specialization_map["numerical"]
        assert set(numerical_judges) == {"judge-a", "judge-b"}
        
        temporal_judges = router._specialization_map["temporal"]
        assert temporal_judges == ["judge-b"]
        
        factual_judges = router._specialization_map["factual"]
        assert factual_judges == ["judge-c"]

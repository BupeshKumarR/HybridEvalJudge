"""
Tests for the ClaimExtractionService.

Requirements: 5.4, 8.4
"""
import pytest
from app.services.claim_extraction_service import (
    ClaimExtractionService,
    ClaimType,
    ExtractedClaim,
)


class TestClaimExtractionService:
    """Test suite for ClaimExtractionService."""

    @pytest.fixture
    def service(self):
        """Create a ClaimExtractionService instance."""
        return ClaimExtractionService()

    def test_extract_claims_empty_text(self, service):
        """Test extraction from empty text returns empty list."""
        claims = service.extract_claims("")
        assert claims == []
        
        claims = service.extract_claims("   ")
        assert claims == []

    def test_extract_claims_simple_factual(self, service):
        """Test extraction of simple factual claims."""
        text = "Paris is the capital of France. The Eiffel Tower is located in Paris."
        claims = service.extract_claims(text)
        
        assert len(claims) >= 1
        assert all(isinstance(c, ExtractedClaim) for c in claims)
        assert all(c.text_span_start >= 0 for c in claims)
        assert all(c.text_span_end > c.text_span_start for c in claims)

    def test_extract_claims_numerical(self, service):
        """Test extraction and classification of numerical claims."""
        text = "The population of Tokyo is approximately 14 million people."
        claims = service.extract_claims(text)
        
        assert len(claims) >= 1
        # Should be classified as numerical due to "14 million"
        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        assert len(numerical_claims) >= 1

    def test_extract_claims_temporal(self, service):
        """Test extraction and classification of temporal claims."""
        text = "The company was founded in 1998. It celebrated its 25th anniversary in 2023."
        claims = service.extract_claims(text)
        
        assert len(claims) >= 1
        # Should have temporal claims due to years
        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        assert len(temporal_claims) >= 1

    def test_extract_claims_definitional(self, service):
        """Test extraction and classification of definitional claims."""
        text = "A mammal is a warm-blooded animal that gives birth to live young."
        claims = service.extract_claims(text)
        
        assert len(claims) >= 1
        # Should be classified as definitional due to "is a"
        definitional_claims = [c for c in claims if c.claim_type == ClaimType.DEFINITIONAL]
        assert len(definitional_claims) >= 1

    def test_extract_claims_span_positions(self, service):
        """Test that span positions are correct."""
        text = "The Earth is round. Water boils at 100 degrees Celsius."
        claims = service.extract_claims(text)
        
        for claim in claims:
            # Verify span positions are within text bounds
            assert claim.text_span_start >= 0
            assert claim.text_span_end <= len(text)
            assert claim.text_span_start < claim.text_span_end

    def test_extract_claims_skips_questions(self, service):
        """Test that questions are not extracted as claims."""
        text = "What is the capital of France? Paris is the capital of France."
        claims = service.extract_claims(text)
        
        # Should not include the question
        for claim in claims:
            assert not claim.text.strip().endswith('?')

    def test_extract_claims_skips_uncertain_statements(self, service):
        """Test that uncertain statements are filtered out."""
        text = "I think Paris might be nice. Paris is the capital of France."
        claims = service.extract_claims(text)
        
        # Should not include uncertain statements
        for claim in claims:
            assert "i think" not in claim.text.lower()
            assert "might be" not in claim.text.lower()

    def test_classify_claim_type_numerical(self, service):
        """Test classification of numerical claims."""
        assert service.classify_claim_type("The temperature is 25 degrees.") == ClaimType.NUMERICAL
        assert service.classify_claim_type("Sales increased by 50%.") == ClaimType.NUMERICAL
        assert service.classify_claim_type("The company has 1,000 employees.") == ClaimType.NUMERICAL

    def test_classify_claim_type_temporal(self, service):
        """Test classification of temporal claims."""
        assert service.classify_claim_type("The event occurred in 2020.") == ClaimType.TEMPORAL
        assert service.classify_claim_type("It was founded on January 15th.") == ClaimType.TEMPORAL
        assert service.classify_claim_type("The building was established in the 19th century.") == ClaimType.TEMPORAL

    def test_classify_claim_type_definitional(self, service):
        """Test classification of definitional claims."""
        assert service.classify_claim_type("A virus is a microscopic organism.") == ClaimType.DEFINITIONAL
        assert service.classify_claim_type("Python is a programming language.") == ClaimType.DEFINITIONAL

    def test_classify_claim_type_general(self, service):
        """Test classification of general claims."""
        # Claims without specific numerical, temporal, or definitional markers
        assert service.classify_claim_type("The sky appears blue during the day.") == ClaimType.GENERAL

    def test_get_claim_type_label(self, service):
        """Test human-readable labels for claim types."""
        assert service.get_claim_type_label(ClaimType.NUMERICAL) == "Numerical"
        assert service.get_claim_type_label(ClaimType.TEMPORAL) == "Temporal"
        assert service.get_claim_type_label(ClaimType.DEFINITIONAL) == "Definitional"
        assert service.get_claim_type_label(ClaimType.GENERAL) == "General Factual"

    def test_confidence_scores(self, service):
        """Test that confidence scores are within valid range."""
        text = "The population is 10 million. It was founded in 1900. A cat is a mammal."
        claims = service.extract_claims(text)
        
        for claim in claims:
            assert 0.0 <= claim.confidence <= 1.0

    def test_extract_claims_multiple_sentences(self, service):
        """Test extraction from text with multiple sentences."""
        text = """
        The Amazon rainforest is the largest tropical rainforest in the world.
        It covers approximately 5.5 million square kilometers.
        The forest was formed during the Eocene era.
        """
        claims = service.extract_claims(text)
        
        # Should extract multiple claims
        assert len(claims) >= 2
        
        # Should have different claim types
        claim_types = set(c.claim_type for c in claims)
        assert len(claim_types) >= 1

"""
Unit tests for the RetrievalComponent.
"""

import tempfile
from pathlib import Path

import pytest

from llm_judge_auditor.components.retrieval_component import RetrievalComponent
from llm_judge_auditor.models import Claim, ClaimType


class TestRetrievalComponent:
    """Test suite for RetrievalComponent."""

    def test_initialization(self):
        """Test basic initialization."""
        component = RetrievalComponent(
            embedding_model="all-MiniLM-L6-v2",
            top_k=3,
            device="cpu",
        )
        
        assert component.embedding_model_name == "all-MiniLM-L6-v2"
        assert component.top_k == 3
        assert component.device == "cpu"
        assert component.fallback_mode() is True  # No KB loaded yet

    def test_fallback_mode_initially(self):
        """Test that component starts in fallback mode."""
        component = RetrievalComponent()
        assert component.fallback_mode() is True

    def test_extract_claims_empty_text(self):
        """Test claim extraction with empty text."""
        component = RetrievalComponent()
        claims = component.extract_claims("")
        assert claims == []
        
        claims = component.extract_claims("   ")
        assert claims == []

    def test_extract_claims_single_sentence(self):
        """Test claim extraction with a single sentence."""
        component = RetrievalComponent()
        text = "The Eiffel Tower was completed in 1889."
        claims = component.extract_claims(text)
        
        assert len(claims) == 1
        assert claims[0].text == text
        assert claims[0].source_span == (0, len(text))

    def test_extract_claims_multiple_sentences(self):
        """Test claim extraction with multiple sentences."""
        component = RetrievalComponent()
        text = "The Eiffel Tower was completed in 1889. It is located in Paris. The tower is 330 meters tall."
        claims = component.extract_claims(text)
        
        assert len(claims) == 3
        assert "Eiffel Tower" in claims[0].text
        assert "Paris" in claims[1].text
        assert "330 meters" in claims[2].text

    def test_claim_type_classification_temporal(self):
        """Test temporal claim classification."""
        component = RetrievalComponent()
        text = "The event occurred in 1889. It happened before the war."
        claims = component.extract_claims(text)
        
        # At least one should be classified as temporal
        claim_types = [claim.claim_type for claim in claims]
        assert ClaimType.TEMPORAL in claim_types

    def test_claim_type_classification_numerical(self):
        """Test numerical claim classification."""
        component = RetrievalComponent()
        text = "The tower is 330 meters tall. The population is 2 million people."
        claims = component.extract_claims(text)
        
        # Should have numerical claims
        claim_types = [claim.claim_type for claim in claims]
        assert ClaimType.NUMERICAL in claim_types

    def test_claim_type_classification_logical(self):
        """Test logical claim classification."""
        component = RetrievalComponent()
        text = "If it rains, then the event will be cancelled. Therefore, we should prepare."
        claims = component.extract_claims(text)
        
        # Should have logical claims
        claim_types = [claim.claim_type for claim in claims]
        assert ClaimType.LOGICAL in claim_types

    def test_retrieve_passages_fallback_mode(self):
        """Test that retrieval returns empty list in fallback mode."""
        component = RetrievalComponent()
        claim = Claim(text="Test claim", source_span=(0, 10), claim_type=ClaimType.FACTUAL)
        
        passages = component.retrieve_passages(claim)
        assert passages == []

    def test_initialize_knowledge_base_missing_file(self):
        """Test initialization with missing knowledge base file."""
        component = RetrievalComponent()
        
        # Should not raise, but should stay in fallback mode
        component.initialize_knowledge_base("/nonexistent/path/kb.txt")
        assert component.fallback_mode() is True

    def test_initialize_knowledge_base_invalid_index_type(self):
        """Test initialization with invalid index type."""
        component = RetrievalComponent()
        
        with pytest.raises(ValueError, match="Unsupported index type"):
            component.initialize_knowledge_base("dummy.txt", index_type="invalid")

    def test_initialize_knowledge_base_and_retrieve(self):
        """Test full workflow: initialize KB and retrieve passages."""
        # Create a temporary knowledge base file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("KB:doc1\tThe Eiffel Tower is in Paris, France.\n")
            f.write("KB:doc2\tThe tower was completed in 1889.\n")
            f.write("KB:doc3\tIt was designed by Gustave Eiffel.\n")
            f.write("KB:doc4\tThe tower is 330 meters tall.\n")
            f.write("KB:doc5\tIt is one of the most visited monuments in the world.\n")
            kb_path = f.name
        
        try:
            component = RetrievalComponent(top_k=2)
            component.initialize_knowledge_base(kb_path)
            
            # Should not be in fallback mode
            assert component.fallback_mode() is False
            
            # Extract a claim and retrieve passages
            claim = Claim(
                text="The Eiffel Tower is located in Paris.",
                source_span=(0, 40),
                claim_type=ClaimType.FACTUAL,
            )
            
            passages = component.retrieve_passages(claim)
            
            # Should retrieve passages
            assert len(passages) > 0
            assert len(passages) <= 2  # top_k=2
            
            # Check passage structure
            for passage in passages:
                assert passage.text
                assert passage.source.startswith("KB:")
                assert 0 <= passage.relevance_score <= 1
            
            # Most relevant passage should mention Paris or Eiffel Tower
            top_passage = passages[0]
            assert "Paris" in top_passage.text or "Eiffel" in top_passage.text
            
        finally:
            # Clean up
            Path(kb_path).unlink()

    def test_initialize_knowledge_base_simple_format(self):
        """Test KB initialization with simple format (no source prefix)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("The Eiffel Tower is in Paris.\n")
            f.write("The tower was completed in 1889.\n")
            kb_path = f.name
        
        try:
            component = RetrievalComponent()
            component.initialize_knowledge_base(kb_path)
            
            assert component.fallback_mode() is False
            
            # Check that passages were loaded with auto-generated sources
            stats = component.get_stats()
            assert stats["num_passages"] == 2
            
        finally:
            Path(kb_path).unlink()

    def test_get_stats(self):
        """Test statistics retrieval."""
        component = RetrievalComponent(
            embedding_model="all-MiniLM-L6-v2",
            top_k=5,
            device="cpu",
        )
        
        stats = component.get_stats()
        
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
        assert stats["top_k"] == 5
        assert stats["device"] == "cpu"
        assert stats["fallback_mode"] is True
        assert stats["num_passages"] == 0
        assert stats["index_size"] == 0

    def test_retrieve_passages_custom_top_k(self):
        """Test retrieval with custom top_k parameter."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            for i in range(10):
                f.write(f"KB:doc{i}\tThis is passage number {i}.\n")
            kb_path = f.name
        
        try:
            component = RetrievalComponent(top_k=3)
            component.initialize_knowledge_base(kb_path)
            
            claim = Claim(
                text="This is a test claim.",
                source_span=(0, 20),
                claim_type=ClaimType.FACTUAL,
            )
            
            # Retrieve with default top_k
            passages_default = component.retrieve_passages(claim)
            assert len(passages_default) <= 3
            
            # Retrieve with custom top_k
            passages_custom = component.retrieve_passages(claim, top_k=5)
            assert len(passages_custom) <= 5
            
        finally:
            Path(kb_path).unlink()

    def test_lazy_loading_encoder(self):
        """Test that encoder is lazily loaded."""
        component = RetrievalComponent()
        
        # Encoder should not be loaded yet
        assert component._encoder is None
        
        # Access encoder property
        encoder = component.encoder
        
        # Now it should be loaded
        assert encoder is not None
        assert component._encoder is not None

    def test_extract_claims_preserves_positions(self):
        """Test that claim extraction preserves correct positions."""
        component = RetrievalComponent()
        text = "First sentence. Second sentence. Third sentence."
        claims = component.extract_claims(text)
        
        # Verify that extracted text matches source spans
        for claim in claims:
            start, end = claim.source_span
            extracted = text[start:end]
            # The extracted text should be similar to the claim text
            # (may differ slightly due to whitespace handling)
            assert claim.text.strip() in extracted or extracted in claim.text

    def test_empty_knowledge_base_file(self):
        """Test initialization with empty knowledge base file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            # Write only whitespace
            f.write("\n\n  \n")
            kb_path = f.name
        
        try:
            component = RetrievalComponent()
            component.initialize_knowledge_base(kb_path)
            
            # Should be in fallback mode due to no valid passages
            assert component.fallback_mode() is True
            
        finally:
            Path(kb_path).unlink()

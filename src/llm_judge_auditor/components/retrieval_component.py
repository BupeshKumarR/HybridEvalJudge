"""
Retrieval Component for knowledge-augmented verification.

This module provides claim extraction and passage retrieval functionality
using sentence transformers and FAISS for efficient similarity search.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models import Claim, ClaimType, Passage

logger = logging.getLogger(__name__)


class RetrievalComponent:
    """
    Component for extracting claims and retrieving relevant passages from knowledge bases.
    
    Supports:
    - Basic claim extraction using sentence splitting
    - Passage retrieval with sentence transformers
    - FAISS-based similarity search
    - Zero-retrieval fallback mode
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        device: Optional[str] = None,
    ):
        """
        Initialize the retrieval component.

        Args:
            embedding_model: Name of the sentence transformer model
            top_k: Number of passages to retrieve per claim
            device: Device to use for embeddings ('cpu', 'cuda', 'mps')
        """
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.device = device or "cpu"
        
        # Lazy loading
        self._encoder: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._passages: List[str] = []
        self._passage_sources: List[str] = []
        self._fallback_mode = True  # Start in fallback mode until KB is loaded
        
        logger.info(
            f"Initialized RetrievalComponent with model={embedding_model}, "
            f"top_k={top_k}, device={self.device}"
        )

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._encoder is None:
            logger.info(f"Loading sentence transformer: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._encoder

    def initialize_knowledge_base(
        self,
        kb_path: str,
        index_type: str = "faiss",
    ) -> None:
        """
        Initialize the knowledge base from a file.

        Args:
            kb_path: Path to knowledge base file (text file with one passage per line)
            index_type: Type of index to use (currently only 'faiss' supported)

        Raises:
            FileNotFoundError: If knowledge base file doesn't exist
            ValueError: If index_type is not supported
        """
        if index_type != "faiss":
            raise ValueError(f"Unsupported index type: {index_type}. Only 'faiss' is supported.")
        
        kb_file = Path(kb_path)
        if not kb_file.exists():
            logger.warning(f"Knowledge base not found at {kb_path}. Using fallback mode.")
            self._fallback_mode = True
            return
        
        logger.info(f"Loading knowledge base from {kb_path}")
        
        # Load passages from file
        with open(kb_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Parse passages (format: "source\ttext" or just "text")
        self._passages = []
        self._passage_sources = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "\t" in line:
                source, text = line.split("\t", 1)
                self._passages.append(text)
                self._passage_sources.append(source)
            else:
                self._passages.append(line)
                self._passage_sources.append(f"KB:passage_{len(self._passages)}")
        
        if not self._passages:
            logger.warning("No passages found in knowledge base. Using fallback mode.")
            self._fallback_mode = True
            return
        
        logger.info(f"Loaded {len(self._passages)} passages from knowledge base")
        
        # Create embeddings for all passages
        logger.info("Creating embeddings for knowledge base...")
        passage_embeddings = self.encoder.encode(
            self._passages,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        
        # Build FAISS index
        dimension = passage_embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(passage_embeddings.astype(np.float32))
        
        logger.info(f"Built FAISS index with {self._index.ntotal} vectors")
        self._fallback_mode = False

    def extract_claims(self, text: str) -> List[Claim]:
        """
        Extract claims from text using sentence splitting.

        Args:
            text: Input text to extract claims from

        Returns:
            List of Claim objects
        """
        if not text or not text.strip():
            return []
        
        # Simple sentence splitting using regex
        # Split on sentence boundaries (., !, ?)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        claims = []
        current_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Find the position of this sentence in the original text
            start_pos = text.find(sentence, current_pos)
            if start_pos == -1:
                # Fallback: use current position
                start_pos = current_pos
            
            end_pos = start_pos + len(sentence)
            
            # Classify claim type based on simple heuristics
            claim_type = self._classify_claim(sentence)
            
            claim = Claim(
                text=sentence,
                source_span=(start_pos, end_pos),
                claim_type=claim_type,
            )
            claims.append(claim)
            
            current_pos = end_pos
        
        logger.debug(f"Extracted {len(claims)} claims from text")
        return claims

    def _classify_claim(self, text: str) -> ClaimType:
        """
        Classify claim type using simple heuristics.

        Args:
            text: Claim text

        Returns:
            ClaimType enum value
        """
        text_lower = text.lower()
        
        # Check for temporal indicators
        temporal_keywords = [
            "year", "date", "time", "when", "before", "after",
            "during", "century", "decade", "ago", "since",
        ]
        if any(keyword in text_lower for keyword in temporal_keywords):
            return ClaimType.TEMPORAL
        
        # Check for numerical indicators
        if re.search(r'\d+', text):
            numerical_keywords = ["number", "amount", "percent", "million", "billion"]
            if any(keyword in text_lower for keyword in numerical_keywords):
                return ClaimType.NUMERICAL
        
        # Check for logical indicators
        logical_keywords = ["if", "then", "because", "therefore", "thus", "hence"]
        if any(keyword in text_lower for keyword in logical_keywords):
            return ClaimType.LOGICAL
        
        # Default to factual
        return ClaimType.FACTUAL

    def retrieve_passages(self, claim: Claim, top_k: Optional[int] = None) -> List[Passage]:
        """
        Retrieve relevant passages for a claim.

        Args:
            claim: Claim to retrieve passages for
            top_k: Number of passages to retrieve (uses self.top_k if None)

        Returns:
            List of Passage objects, empty if in fallback mode
        """
        if self._fallback_mode or self._index is None:
            logger.debug("In fallback mode, no passages retrieved")
            return []
        
        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self._passages))  # Don't retrieve more than available
        
        if k == 0:
            return []
        
        # Encode the claim
        claim_embedding = self.encoder.encode(
            [claim.text],
            convert_to_numpy=True,
        )
        
        # Search the index
        distances, indices = self._index.search(claim_embedding.astype(np.float32), k)
        
        # Convert to Passage objects
        passages = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._passages):
                # Convert L2 distance to similarity score (inverse)
                # Lower distance = higher similarity
                relevance_score = 1.0 / (1.0 + float(dist))
                
                passage = Passage(
                    text=self._passages[idx],
                    source=self._passage_sources[idx],
                    relevance_score=relevance_score,
                )
                passages.append(passage)
        
        logger.debug(f"Retrieved {len(passages)} passages for claim")
        return passages

    def fallback_mode(self) -> bool:
        """
        Check if component is in zero-retrieval fallback mode.

        Returns:
            True if in fallback mode (no KB loaded), False otherwise
        """
        return self._fallback_mode

    def get_stats(self) -> dict:
        """
        Get statistics about the retrieval component.

        Returns:
            Dictionary with component statistics
        """
        return {
            "embedding_model": self.embedding_model_name,
            "top_k": self.top_k,
            "device": self.device,
            "fallback_mode": self._fallback_mode,
            "num_passages": len(self._passages),
            "index_size": self._index.ntotal if self._index else 0,
        }

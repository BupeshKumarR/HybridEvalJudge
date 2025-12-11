"""
Claim Extraction Service for extracting and classifying factual claims from LLM responses.

This service extracts factual claims from text, classifies them by type
(numerical, temporal, definitional, general), and returns claim text with span positions.

Requirements: 5.4, 8.4
"""
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Enable detailed logging for debugging evaluation issues
DETAILED_LOGGING = True


class ClaimType(str, Enum):
    """Types of claims for specialized routing and display."""
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    DEFINITIONAL = "definitional"
    GENERAL = "general"


@dataclass
class ExtractedClaim:
    """
    A single claim extracted from LLM response.
    
    Attributes:
        text: The claim text
        claim_type: Type of claim (numerical, temporal, definitional, general)
        text_span_start: Start position in original text
        text_span_end: End position in original text
        confidence: Confidence score for the extraction (0.0 to 1.0)
    """
    text: str
    claim_type: ClaimType
    text_span_start: int
    text_span_end: int
    confidence: float = 0.8


class ClaimExtractionService:
    """
    Service for extracting factual claims from LLM responses.
    
    This service:
    1. Extracts factual claims from text
    2. Classifies claim types (numerical, temporal, definitional, general)
    3. Returns claim text with span positions for highlighting
    
    Requirements: 5.4, 8.4
    """
    
    def __init__(self):
        """Initialize the ClaimExtractionService."""
        logger.info("ClaimExtractionService initialized")
    
    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract factual claims from text.
        
        This method splits text into sentences and identifies those that
        contain factual claims worth verifying.
        
        Args:
            text: The LLM response text to extract claims from
            
        Returns:
            List of ExtractedClaim objects with text, type, and span positions
        """
        if DETAILED_LOGGING:
            logger.info(f"[CLAIM_EXTRACTION] Starting extraction from text of length {len(text) if text else 0}")
            logger.info(f"[CLAIM_EXTRACTION] Text preview: {text[:200] if text else 'EMPTY'}...")
        
        if not text or not text.strip():
            logger.warning("[CLAIM_EXTRACTION] Empty or whitespace-only text provided")
            return []
        
        claims = []
        sentences = self._split_into_sentences(text)
        
        if DETAILED_LOGGING:
            logger.info(f"[CLAIM_EXTRACTION] Split into {len(sentences)} sentences")
        
        skipped_short = 0
        skipped_non_factual = 0
        
        for sentence, start_pos, end_pos in sentences:
            sentence_stripped = sentence.strip()
            
            # More lenient minimum length - reduced from 10 to 5 characters
            if len(sentence_stripped) < 5:
                skipped_short += 1
                if DETAILED_LOGGING:
                    logger.debug(f"[CLAIM_EXTRACTION] Skipped short sentence: '{sentence_stripped}'")
                continue
            
            # Check if sentence contains a factual claim
            is_factual = self._is_factual_claim(sentence)
            if DETAILED_LOGGING:
                logger.debug(f"[CLAIM_EXTRACTION] Sentence '{sentence_stripped[:50]}...' is_factual={is_factual}")
            
            if is_factual:
                claim_type = self._classify_claim(sentence)
                confidence = self._calculate_confidence(sentence, claim_type)
                
                claim = ExtractedClaim(
                    text=sentence_stripped,
                    claim_type=claim_type,
                    text_span_start=start_pos,
                    text_span_end=end_pos,
                    confidence=confidence
                )
                claims.append(claim)
                if DETAILED_LOGGING:
                    logger.info(f"[CLAIM_EXTRACTION] Extracted claim: type={claim_type.value}, text='{sentence_stripped[:50]}...'")
            else:
                skipped_non_factual += 1
        
        # FALLBACK: If no claims extracted but text has content, treat entire response as one claim
        # This ensures evaluation can proceed even with short/simple responses from older models
        if not claims and len(text.strip()) >= 5:
            logger.warning(f"[CLAIM_EXTRACTION] No claims extracted from {len(sentences)} sentences. Using fallback.")
            fallback_claim = ExtractedClaim(
                text=text.strip()[:500],  # Limit to 500 chars
                claim_type=ClaimType.GENERAL,
                text_span_start=0,
                text_span_end=min(len(text), 500),
                confidence=0.5  # Lower confidence for fallback
            )
            claims.append(fallback_claim)
            logger.info(f"[CLAIM_EXTRACTION] Created fallback claim from entire response")
        
        logger.info(
            f"[CLAIM_EXTRACTION] Extraction complete: {len(claims)} claims extracted, "
            f"{skipped_short} skipped (too short), {skipped_non_factual} skipped (non-factual)"
        )
        return claims
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with their span positions.
        
        Args:
            text: Text to split
            
        Returns:
            List of tuples (sentence, start_pos, end_pos)
        """
        sentences = []
        
        # Common abbreviations to avoid splitting on
        abbreviations = {'dr', 'mr', 'mrs', 'ms', 'prof', 'jr', 'sr', 'vs', 'etc', 'e.g', 'i.e'}
        
        # Simple sentence splitting approach
        # Split on sentence-ending punctuation followed by space and capital letter
        # or end of string
        current_start = 0
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Check for sentence-ending punctuation
            if char in '.!?':
                # Check if this is likely a sentence boundary
                is_boundary = False
                
                # Check what follows
                if i + 1 >= len(text):
                    # End of text
                    is_boundary = True
                elif i + 1 < len(text) and text[i + 1] in ' \n\t':
                    # Followed by whitespace
                    # Check if next non-whitespace is uppercase or end
                    j = i + 1
                    while j < len(text) and text[j] in ' \n\t':
                        j += 1
                    if j >= len(text) or text[j].isupper():
                        is_boundary = True
                
                # Check for abbreviations (only for periods)
                if char == '.' and is_boundary:
                    # Look back to find the word before the period
                    word_start = i - 1
                    while word_start >= 0 and text[word_start].isalpha():
                        word_start -= 1
                    word_start += 1
                    word_before = text[word_start:i].lower()
                    
                    if word_before in abbreviations:
                        is_boundary = False
                
                if is_boundary:
                    # Extract the sentence
                    sentence_text = text[current_start:i + 1].strip()
                    if sentence_text:
                        sentences.append((sentence_text, current_start, i + 1))
                    
                    # Move to next sentence start
                    current_start = i + 1
                    while current_start < len(text) and text[current_start] in ' \n\t':
                        current_start += 1
            
            i += 1
        
        # Handle any remaining text
        if current_start < len(text):
            remaining = text[current_start:].strip()
            if remaining:
                sentences.append((remaining, current_start, len(text)))
        
        return sentences
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence contains a factual claim.
        
        Made more lenient to handle varied response formats from different models.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            True if the sentence contains a factual claim
        """
        sentence_lower = sentence.lower().strip()
        
        # Skip questions
        if sentence.strip().endswith('?'):
            if DETAILED_LOGGING:
                logger.debug(f"[CLAIM_EXTRACTION] Skipping question: '{sentence[:30]}...'")
            return False
        
        # Skip imperative sentences (commands) - but only if they START with these
        imperative_starters = ['please', 'let me', 'let\'s', 'try to', 'make sure']
        if any(sentence_lower.startswith(starter) for starter in imperative_starters):
            if DETAILED_LOGGING:
                logger.debug(f"[CLAIM_EXTRACTION] Skipping imperative: '{sentence[:30]}...'")
            return False
        
        # Skip hedged/uncertain statements
        uncertainty_markers = [
            'i think', 'i believe', 'maybe', 'perhaps', 'possibly',
            'might be', 'could be', 'i\'m not sure', 'i don\'t know',
            'i cannot', 'i can\'t'
        ]
        if any(marker in sentence_lower for marker in uncertainty_markers):
            if DETAILED_LOGGING:
                logger.debug(f"[CLAIM_EXTRACTION] Skipping uncertain: '{sentence[:30]}...'")
            return False
        
        # Look for factual indicators - expanded list for better coverage
        factual_indicators = [
            # Definitional patterns
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bhas\b', r'\bhave\b', r'\bhad\b',
            r'\bcan\b', r'\bcould\b', r'\bwill\b', r'\bwould\b',
            # Numerical patterns
            r'\d+', r'percent', r'%',
            # Temporal patterns
            r'\b(in|during|on|at)\s+\d{4}\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            # Factual verbs
            r'\b(contains?|consists?|includes?|comprises?)\b',
            r'\b(located|founded|established|created|invented)\b',
            r'\b(means?|refers?\s+to|called|known\s+as)\b',
            # Common factual patterns
            r'\b(because|since|therefore|thus|hence)\b',
            r'\b(first|second|third|finally|also|additionally)\b',
            r'\b(example|such\s+as|like|including)\b',
        ]
        
        for pattern in factual_indicators:
            if re.search(pattern, sentence_lower):
                return True
        
        # More lenient default: consider sentences with 3+ words as potential claims
        # (reduced from 5 to handle shorter model responses)
        word_count = len(sentence.split())
        if word_count >= 3:
            if DETAILED_LOGGING:
                logger.debug(f"[CLAIM_EXTRACTION] Accepting by word count ({word_count}): '{sentence[:30]}...'")
            return True
        
        return False
    
    def _classify_claim(self, sentence: str) -> ClaimType:
        """
        Classify a claim by its type.
        
        Args:
            sentence: Sentence to classify
            
        Returns:
            ClaimType enum value
        """
        sentence_lower = sentence.lower()
        
        # Check for numerical claims
        numerical_patterns = [
            r'\d+\.?\d*\s*%',  # Percentages
            r'\d+\.?\d*\s*(million|billion|thousand|hundred)',  # Large numbers
            r'\d+\.?\d*\s*(kg|km|meters?|cm|mm|lb|oz|ft|in|degrees?)',  # Measurements
            r'\d+\.?\d*\s*(dollars?|euros?|pounds?|yen|\$|€|£)',  # Currency
            r'\b\d{1,3}(,\d{3})+\b',  # Numbers with commas
            r'\b\d+\s*(years?|months?|days?|hours?|minutes?|seconds?)\b',  # Time durations
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, sentence_lower):
                return ClaimType.NUMERICAL
        
        # Check for temporal claims
        temporal_patterns = [
            r'\b(in|during|on|at|since|until|before|after)\s+\d{4}\b',  # Years with prepositions
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(today|yesterday|tomorrow|last\s+week|next\s+week)\b',
            r'\b\d{1,2}(st|nd|rd|th)\s+century\b',  # Centuries
            r'\b(ancient|medieval|modern|contemporary)\b',  # Historical periods
            r'\b(founded|established|created|invented|discovered)\s+in\b',
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, sentence_lower):
                return ClaimType.TEMPORAL
        
        # Check for definitional claims
        definitional_patterns = [
            r'\bis\s+(a|an|the)\s+\w+',  # "X is a/an/the Y"
            r'\bare\s+(a|an|the)?\s*\w+s?\b',  # "X are Y"
            r'\b(defined\s+as|refers?\s+to|means?)\b',
            r'\b(known\s+as|called|named)\b',
            r'\b(type|kind|form|category)\s+of\b',
            r'\b(consists?\s+of|comprises?|includes?)\b',
        ]
        
        for pattern in definitional_patterns:
            if re.search(pattern, sentence_lower):
                return ClaimType.DEFINITIONAL
        
        # Default to general factual claim
        return ClaimType.GENERAL
    
    def _calculate_confidence(self, sentence: str, claim_type: ClaimType) -> float:
        """
        Calculate confidence score for the claim extraction.
        
        Args:
            sentence: The claim sentence
            claim_type: The classified claim type
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7
        
        # Boost confidence for specific claim types
        if claim_type == ClaimType.NUMERICAL:
            # Numerical claims are usually clear-cut
            base_confidence += 0.15
        elif claim_type == ClaimType.TEMPORAL:
            # Temporal claims are also fairly clear
            base_confidence += 0.1
        elif claim_type == ClaimType.DEFINITIONAL:
            # Definitional claims are moderately clear
            base_confidence += 0.05
        
        # Adjust based on sentence characteristics
        word_count = len(sentence.split())
        
        # Very short sentences might be incomplete
        if word_count < 5:
            base_confidence -= 0.1
        # Medium length sentences are ideal
        elif 10 <= word_count <= 25:
            base_confidence += 0.05
        # Very long sentences might contain multiple claims
        elif word_count > 40:
            base_confidence -= 0.05
        
        # Ensure confidence is within bounds
        return max(0.5, min(0.95, base_confidence))
    
    def classify_claim_type(self, claim_text: str) -> ClaimType:
        """
        Public method to classify a single claim's type.
        
        Args:
            claim_text: The claim text to classify
            
        Returns:
            ClaimType enum value
        """
        return self._classify_claim(claim_text)
    
    def get_claim_type_label(self, claim_type: ClaimType) -> str:
        """
        Get a human-readable label for a claim type.
        
        Args:
            claim_type: The ClaimType enum value
            
        Returns:
            Human-readable label string
        """
        labels = {
            ClaimType.NUMERICAL: "Numerical",
            ClaimType.TEMPORAL: "Temporal",
            ClaimType.DEFINITIONAL: "Definitional",
            ClaimType.GENERAL: "General Factual"
        }
        return labels.get(claim_type, "Unknown")

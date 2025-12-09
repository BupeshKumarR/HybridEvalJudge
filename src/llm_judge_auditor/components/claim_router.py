"""
Claim Router for specialized judge selection.

This module provides the ClaimRouter class for classifying claims by type
and routing them to specialized judges based on their expertise areas.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from llm_judge_auditor.models import Claim, ClaimType

logger = logging.getLogger(__name__)


class ClaimRouter:
    """
    Routes claims to specialized judges based on claim type classification.

    This class analyzes claims to determine their type (numerical, temporal,
    factual, logical, commonsense) and routes them to judges that specialize
    in evaluating those types of claims.

    Requirements: 2.3 (specialized judge selection)
    """

    def __init__(self, judge_specializations: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the ClaimRouter.

        Args:
            judge_specializations: Optional mapping of judge names to their
                specialization areas. Format:
                {
                    "judge_name": ["numerical", "temporal"],
                    "another_judge": ["factual", "logical"]
                }
                If None, all judges are considered general-purpose.

        Example:
            >>> specializations = {
            ...     "llama-3-8b": ["factual", "logical"],
            ...     "mistral-7b": ["numerical", "temporal"],
            ...     "phi-3-mini": ["commonsense"]
            ... }
            >>> router = ClaimRouter(specializations)
        """
        self.judge_specializations = judge_specializations or {}
        
        # Build reverse mapping: claim_type -> list of specialized judges
        self._specialization_map: Dict[str, List[str]] = {}
        for judge_name, specializations in self.judge_specializations.items():
            for spec in specializations:
                if spec not in self._specialization_map:
                    self._specialization_map[spec] = []
                self._specialization_map[spec].append(judge_name)

        logger.info(
            f"ClaimRouter initialized with {len(self.judge_specializations)} "
            f"specialized judges"
        )

    def classify_claim(self, claim: Claim) -> ClaimType:
        """
        Classify a claim by its type.

        This method analyzes the claim text to determine whether it is:
        - NUMERICAL: Contains numbers, quantities, measurements
        - TEMPORAL: Contains dates, times, temporal references
        - LOGICAL: Contains logical reasoning, conditionals, causation
        - COMMONSENSE: Requires common sense reasoning
        - FACTUAL: General factual statement (default)

        Args:
            claim: Claim object to classify

        Returns:
            ClaimType enum value

        Example:
            >>> claim = Claim(text="The temperature was 25 degrees.", source_span=(0, 30))
            >>> router = ClaimRouter()
            >>> claim_type = router.classify_claim(claim)
            >>> print(claim_type)
            ClaimType.NUMERICAL
        """
        text = claim.text.lower()

        # Check for logical claims FIRST (before temporal)
        # Look for logical connectives, conditionals, causation
        # These are strong indicators and should take precedence
        strong_logical_patterns = [
            r'\b(if|then)\b.*\b(then|will|would)\b',  # if-then constructs
            r'\b(therefore|thus|hence|consequently)\b',  # Strong logical connectives
            r'\b(implies|entails|follows from)\b',  # Logical relationships
        ]
        
        for pattern in strong_logical_patterns:
            if re.search(pattern, text):
                logger.debug(f"Classified as LOGICAL: {claim.text[:50]}...")
                return ClaimType.LOGICAL

        # Check for numerical claims with specific units/contexts
        # Prioritize specific numerical patterns over generic numbers
        specific_numerical_patterns = [
            r'\d+\.?\d*\s*%',  # Percentages: 50%, 12.5%
            r'\d+\.?\d*\s*(million|billion|thousand|hundred)',  # Large numbers
            r'\d+\.?\d*\s*(kg|km|meters?|cm|mm|lb|oz|ft|in|degrees?)',  # Measurements
            r'\d+\.?\d*\s*(dollars?|euros?|pounds?|yen)',  # Currency
        ]
        
        for pattern in specific_numerical_patterns:
            if re.search(pattern, text):
                logger.debug(f"Classified as NUMERICAL: {claim.text[:50]}...")
                return ClaimType.NUMERICAL

        # Check for temporal claims
        # Look for dates, times, temporal references
        # But exclude bare years that might be part of other contexts
        temporal_patterns = [
            r'\b(in|during|on|at)\s+\d{4}\b',  # Years with prepositions: "in 1989"
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(century|decade|year|month|week|day|hour|minute|second)s?\b',
            r'\b(ago|since|until)\b',
            r'\d{1,2}:\d{2}',  # Times: 10:30, 3:45
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, text):
                logger.debug(f"Classified as TEMPORAL: {claim.text[:50]}...")
                return ClaimType.TEMPORAL

        # Check for commonsense claims
        # Look for everyday concepts, human behavior, physical properties
        commonsense_patterns = [
            r'\b(people|person|humans?|everyone|someone)\b',
            r'\b(usually|typically|generally|normally|commonly)\b',
            r'\b(feel|think|believe|prefer)\b',
            r'\b(happier|sadder|better|worse)\b.*\b(when|if)\b',  # Emotional states with conditions
        ]
        
        commonsense_count = sum(1 for pattern in commonsense_patterns if re.search(pattern, text))
        if commonsense_count >= 2:  # Require at least 2 commonsense indicators
            logger.debug(f"Classified as COMMONSENSE: {claim.text[:50]}...")
            return ClaimType.COMMONSENSE

        # Check for weaker logical patterns (after other checks)
        weak_logical_patterns = [
            r'\b(because|since)\b',  # Causal relationships
            r'\b(all|some|none|every)\b',  # Quantifiers
        ]
        
        logical_count = sum(1 for pattern in weak_logical_patterns if re.search(pattern, text))
        if logical_count >= 1:
            logger.debug(f"Classified as LOGICAL: {claim.text[:50]}...")
            return ClaimType.LOGICAL

        # Check for generic numbers last (lowest priority)
        if re.search(r'\b\d+\b', text):
            logger.debug(f"Classified as NUMERICAL: {claim.text[:50]}...")
            return ClaimType.NUMERICAL

        # Default to factual
        logger.debug(f"Classified as FACTUAL (default): {claim.text[:50]}...")
        return ClaimType.FACTUAL

    def route_to_judge(
        self,
        claim: Claim,
        available_judges: List[str],
    ) -> str:
        """
        Route a claim to the most appropriate judge based on specialization.

        This method:
        1. Classifies the claim if not already classified
        2. Finds judges specialized in that claim type
        3. Returns the best match from available judges
        4. Falls back to first available judge if no specialist found

        Args:
            claim: Claim to route
            available_judges: List of judge names that are currently available

        Returns:
            Name of the selected judge

        Raises:
            ValueError: If no judges are available

        Example:
            >>> claim = Claim(text="The temperature was 25 degrees.", source_span=(0, 30))
            >>> router = ClaimRouter({
            ...     "mistral-7b": ["numerical", "temporal"],
            ...     "llama-3-8b": ["factual"]
            ... })
            >>> judge = router.route_to_judge(claim, ["mistral-7b", "llama-3-8b"])
            >>> print(judge)
            mistral-7b
        """
        if not available_judges:
            raise ValueError("No judges available for routing")

        # Classify the claim if not already classified
        claim_type = claim.claim_type
        if claim_type == ClaimType.FACTUAL and claim.text:
            # Re-classify to get more specific type
            claim_type = self.classify_claim(claim)
            # Update the claim object
            claim.claim_type = claim_type

        # Get specialized judges for this claim type
        claim_type_str = claim_type.value
        specialized_judges = self._specialization_map.get(claim_type_str, [])

        # Find intersection of specialized judges and available judges
        matching_judges = [j for j in specialized_judges if j in available_judges]

        if matching_judges:
            # Return first matching specialized judge
            selected = matching_judges[0]
            logger.info(
                f"Routed {claim_type_str} claim to specialized judge: {selected}"
            )
            return selected
        else:
            # No specialized judge available, use first available judge
            selected = available_judges[0]
            logger.info(
                f"No specialized judge for {claim_type_str}, using default: {selected}"
            )
            return selected

    def route_claims_to_judges(
        self,
        claims: List[Claim],
        available_judges: List[str],
    ) -> Dict[str, List[Claim]]:
        """
        Route multiple claims to judges, grouping by assigned judge.

        This method processes a batch of claims and groups them by the
        judge they should be evaluated by, enabling efficient batch processing.

        Args:
            claims: List of claims to route
            available_judges: List of judge names that are currently available

        Returns:
            Dictionary mapping judge names to lists of claims assigned to them

        Raises:
            ValueError: If no judges are available

        Example:
            >>> claims = [
            ...     Claim(text="The temperature was 25 degrees.", source_span=(0, 30)),
            ...     Claim(text="Paris is the capital of France.", source_span=(0, 32)),
            ... ]
            >>> router = ClaimRouter({
            ...     "mistral-7b": ["numerical"],
            ...     "llama-3-8b": ["factual"]
            ... })
            >>> routing = router.route_claims_to_judges(claims, ["mistral-7b", "llama-3-8b"])
            >>> for judge, judge_claims in routing.items():
            ...     print(f"{judge}: {len(judge_claims)} claims")
        """
        if not available_judges:
            raise ValueError("No judges available for routing")

        # Initialize routing dictionary
        routing: Dict[str, List[Claim]] = {judge: [] for judge in available_judges}

        # Route each claim
        for claim in claims:
            selected_judge = self.route_to_judge(claim, available_judges)
            routing[selected_judge].append(claim)

        # Log routing summary
        summary = ", ".join(
            f"{judge}: {len(claims)}" for judge, claims in routing.items() if claims
        )
        logger.info(f"Routed {len(claims)} claims to judges: {summary}")

        return routing

    def get_judge_specializations(self, judge_name: str) -> List[str]:
        """
        Get the specialization areas for a specific judge.

        Args:
            judge_name: Name of the judge

        Returns:
            List of specialization areas (claim types)

        Example:
            >>> router = ClaimRouter({"mistral-7b": ["numerical", "temporal"]})
            >>> specs = router.get_judge_specializations("mistral-7b")
            >>> print(specs)
            ['numerical', 'temporal']
        """
        return self.judge_specializations.get(judge_name, [])

    def get_specialized_judges(self, claim_type: ClaimType) -> List[str]:
        """
        Get all judges specialized in a specific claim type.

        Args:
            claim_type: Type of claim

        Returns:
            List of judge names specialized in that claim type

        Example:
            >>> router = ClaimRouter({
            ...     "mistral-7b": ["numerical"],
            ...     "phi-3-mini": ["numerical", "commonsense"]
            ... })
            >>> judges = router.get_specialized_judges(ClaimType.NUMERICAL)
            >>> print(judges)
            ['mistral-7b', 'phi-3-mini']
        """
        return self._specialization_map.get(claim_type.value, [])

    def update_specializations(
        self,
        judge_name: str,
        specializations: List[str],
    ) -> None:
        """
        Update or add specializations for a judge.

        This allows dynamic configuration of judge specializations.

        Args:
            judge_name: Name of the judge
            specializations: List of specialization areas

        Example:
            >>> router = ClaimRouter()
            >>> router.update_specializations("new-judge", ["numerical", "temporal"])
        """
        # Remove old specializations from reverse map
        if judge_name in self.judge_specializations:
            old_specs = self.judge_specializations[judge_name]
            for spec in old_specs:
                if spec in self._specialization_map:
                    self._specialization_map[spec] = [
                        j for j in self._specialization_map[spec] if j != judge_name
                    ]

        # Add new specializations
        self.judge_specializations[judge_name] = specializations
        for spec in specializations:
            if spec not in self._specialization_map:
                self._specialization_map[spec] = []
            if judge_name not in self._specialization_map[spec]:
                self._specialization_map[spec].append(judge_name)

        logger.info(
            f"Updated specializations for {judge_name}: {specializations}"
        )

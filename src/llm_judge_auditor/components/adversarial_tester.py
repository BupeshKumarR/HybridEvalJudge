"""
Adversarial Test Harness for robustness testing of the evaluation toolkit.

This module provides the AdversarialTester class, which generates adversarial
variants of input texts with subtle factual perturbations and tests the
evaluation system's ability to detect these manipulations.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """
    Result from a single perturbation test.
    
    Attributes:
        original_text: The original unperturbed text
        perturbed_text: The text with perturbations applied
        perturbation_type: Type of perturbation applied
        perturbations_applied: List of specific perturbations made
        detected: Whether the evaluation system detected the perturbation
        original_score: Score for the original text
        perturbed_score: Score for the perturbed text
        score_delta: Difference between original and perturbed scores
    """
    original_text: str
    perturbed_text: str
    perturbation_type: str
    perturbations_applied: List[str] = field(default_factory=list)
    detected: bool = False
    original_score: float = 0.0
    perturbed_score: float = 0.0
    score_delta: float = 0.0


@dataclass
class SymmetryReport:
    """
    Report on pairwise ranking symmetry testing.
    
    Attributes:
        candidate_a: First candidate text
        candidate_b: Second candidate text
        source: Source text used for comparison
        ab_winner: Winner when comparing A vs B
        ba_winner: Winner when comparing B vs A
        is_symmetric: Whether rankings are consistent
        ab_reasoning: Reasoning for A vs B comparison
        ba_reasoning: Reasoning for B vs A comparison
    """
    candidate_a: str
    candidate_b: str
    source: str
    ab_winner: str
    ba_winner: str
    is_symmetric: bool
    ab_reasoning: str = ""
    ba_reasoning: str = ""


@dataclass
class RobustnessReport:
    """
    Comprehensive robustness testing report.
    
    Attributes:
        total_tests: Total number of perturbation tests run
        detected_count: Number of perturbations detected
        missed_count: Number of perturbations missed
        detection_rate: Percentage of perturbations detected
        false_positive_rate: Rate of false positives (if applicable)
        perturbation_results: Detailed results for each perturbation
        by_type: Detection rates broken down by perturbation type
        metadata: Additional metadata about the test run
    """
    total_tests: int
    detected_count: int
    missed_count: int
    detection_rate: float
    false_positive_rate: float
    perturbation_results: List[PerturbationResult] = field(default_factory=list)
    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdversarialTester:
    """
    Adversarial test harness for evaluating robustness of the evaluation toolkit.
    
    This class generates adversarial variants of input texts with subtle factual
    perturbations (date shifts, location swaps, number changes) and tests whether
    the evaluation system can detect these manipulations. It also tests pairwise
    ranking symmetry and other robustness properties.
    
    Example:
        >>> from llm_judge_auditor import EvaluationToolkit
        >>> toolkit = EvaluationToolkit.from_preset("balanced")
        >>> tester = AdversarialTester(toolkit)
        >>> 
        >>> # Generate perturbations
        >>> text = "The Eiffel Tower was completed in 1889 in Paris."
        >>> perturbations = tester.generate_perturbations(
        ...     text=text,
        ...     perturbation_types=["date_shift", "location_swap"]
        ... )
        >>> 
        >>> # Test robustness
        >>> source = "The Eiffel Tower was built for the 1889 World's Fair in Paris."
        >>> report = tester.test_robustness(
        ...     source=source,
        ...     original=text,
        ...     perturbations=perturbations
        ... )
        >>> print(f"Detection Rate: {report.detection_rate:.1%}")
    """
    
    def __init__(self, evaluation_toolkit: Any, detection_threshold: float = 10.0):
        """
        Initialize the AdversarialTester.
        
        Args:
            evaluation_toolkit: EvaluationToolkit instance to test
            detection_threshold: Score difference threshold for detecting perturbations
                                (default: 10.0 points)
        """
        self.toolkit = evaluation_toolkit
        self.detection_threshold = detection_threshold
        logger.info(f"Initialized AdversarialTester with threshold={detection_threshold}")
    
    def generate_perturbations(
        self,
        text: str,
        perturbation_types: List[str],
        num_variants: int = 1
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Generate adversarial perturbations of the input text.
        
        Args:
            text: Original text to perturb
            perturbation_types: List of perturbation types to apply
                              (e.g., ["date_shift", "location_swap", "number_change"])
            num_variants: Number of variants to generate per perturbation type
        
        Returns:
            List of tuples (perturbed_text, perturbation_type, perturbations_applied)
        
        Supported perturbation types:
            - "date_shift": Shift dates by small amounts (days, months, years)
            - "location_swap": Replace locations with similar but incorrect ones
            - "number_change": Modify numbers by small amounts
            - "entity_replace": Replace named entities with similar ones
        """
        perturbations = []
        
        for perturbation_type in perturbation_types:
            for _ in range(num_variants):
                if perturbation_type == "date_shift":
                    result = self._perturb_dates(text)
                elif perturbation_type == "location_swap":
                    result = self._perturb_locations(text)
                elif perturbation_type == "number_change":
                    result = self._perturb_numbers(text)
                elif perturbation_type == "entity_replace":
                    result = self._perturb_entities(text)
                else:
                    logger.warning(f"Unknown perturbation type: {perturbation_type}")
                    continue
                
                if result:
                    perturbed_text, changes = result
                    perturbations.append((perturbed_text, perturbation_type, changes))
        
        logger.info(f"Generated {len(perturbations)} perturbations from {len(perturbation_types)} types")
        return perturbations
    
    def _perturb_dates(self, text: str) -> Optional[Tuple[str, List[str]]]:
        """
        Perturb dates in the text by shifting them slightly.
        
        Args:
            text: Text containing dates
        
        Returns:
            Tuple of (perturbed_text, list of changes) or None if no dates found
        """
        # Match various date formats: YYYY, MM/DD/YYYY, Month DD, YYYY, etc.
        date_patterns = [
            (r'\b[12]\d{3}\b', 'year'),  # 4-digit years (1000-2999)
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+[12]\d{3}\b', 'full_date'),
            (r'\b\d{1,2}/\d{1,2}/[12]\d{3}\b', 'numeric_date'),
        ]
        
        changes = []
        perturbed = text
        
        for pattern, date_type in date_patterns:
            matches = list(re.finditer(pattern, perturbed))
            if matches:
                # Pick a random match to perturb
                match = random.choice(matches)
                original = match.group(0)
                
                if date_type == 'year':
                    # Shift year by 1-5 years
                    year = int(original)
                    shift = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                    new_year = str(year + shift)
                    # Replace only the first occurrence
                    start, end = match.span()
                    perturbed = perturbed[:start] + new_year + perturbed[end:]
                    changes.append(f"Changed year {original} to {new_year}")
                    return perturbed, changes
                
                elif date_type == 'full_date':
                    # Extract and shift the year
                    year_match = re.search(r'[12]\d{3}', original)
                    if year_match:
                        year = int(year_match.group(0))
                        shift = random.choice([-3, -2, -1, 1, 2, 3])
                        new_year = str(year + shift)
                        new_date = original.replace(year_match.group(0), new_year)
                        start, end = match.span()
                        perturbed = perturbed[:start] + new_date + perturbed[end:]
                        changes.append(f"Changed date '{original}' to '{new_date}'")
                        return perturbed, changes
                
                elif date_type == 'numeric_date':
                    # Extract and shift the year
                    year_match = re.search(r'[12]\d{3}', original)
                    if year_match:
                        year = int(year_match.group(0))
                        shift = random.choice([-3, -2, -1, 1, 2, 3])
                        new_year = str(year + shift)
                        new_date = original.replace(year_match.group(0), new_year)
                        start, end = match.span()
                        perturbed = perturbed[:start] + new_date + perturbed[end:]
                        changes.append(f"Changed date '{original}' to '{new_date}'")
                        return perturbed, changes
        
        return None
    
    def _perturb_locations(self, text: str) -> Optional[Tuple[str, List[str]]]:
        """
        Perturb location names in the text.
        
        Args:
            text: Text containing location names
        
        Returns:
            Tuple of (perturbed_text, list of changes) or None if no locations found
        """
        # Common location swaps (similar but incorrect)
        location_swaps = {
            'Paris': ['London', 'Berlin', 'Rome', 'Madrid'],
            'London': ['Paris', 'Berlin', 'Amsterdam', 'Brussels'],
            'New York': ['Los Angeles', 'Chicago', 'Boston', 'Philadelphia'],
            'Tokyo': ['Beijing', 'Seoul', 'Shanghai', 'Osaka'],
            'France': ['Germany', 'Italy', 'Spain', 'Belgium'],
            'Germany': ['France', 'Italy', 'Poland', 'Austria'],
            'United States': ['Canada', 'Mexico', 'United Kingdom'],
            'China': ['Japan', 'Korea', 'India', 'Vietnam'],
            'California': ['Texas', 'Florida', 'New York', 'Nevada'],
            'Europe': ['Asia', 'Africa', 'North America'],
        }
        
        changes = []
        perturbed = text
        
        for original_location, alternatives in location_swaps.items():
            if original_location in text:
                new_location = random.choice(alternatives)
                perturbed = text.replace(original_location, new_location, 1)
                changes.append(f"Changed location '{original_location}' to '{new_location}'")
                break
        
        if changes:
            return perturbed, changes
        return None
    
    def _perturb_numbers(self, text: str) -> Optional[Tuple[str, List[str]]]:
        """
        Perturb numbers in the text by changing them slightly.
        
        Args:
            text: Text containing numbers
        
        Returns:
            Tuple of (perturbed_text, list of changes) or None if no numbers found
        """
        # Match numbers (integers and decimals)
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        matches = list(re.finditer(number_pattern, text))
        
        if not matches:
            return None
        
        # Filter out years (4-digit numbers that look like years)
        non_year_matches = [
            m for m in matches
            if not (len(m.group(0)) == 4 and m.group(0).startswith(('19', '20')))
        ]
        
        if not non_year_matches:
            return None
        
        changes = []
        perturbed = text
        
        # Pick a random number to perturb
        match = random.choice(non_year_matches)
        original = match.group(0)
        
        if '.' in original:
            # Decimal number
            num = float(original)
            # Change by 5-20%
            change_pct = random.uniform(0.05, 0.20) * random.choice([-1, 1])
            new_num = num * (1 + change_pct)
            new_str = f"{new_num:.2f}"
        else:
            # Integer
            num = int(original)
            # Change by 1-10 or 5-20% for larger numbers
            if num < 20:
                change = random.randint(1, 5) * random.choice([-1, 1])
            else:
                change = int(num * random.uniform(0.05, 0.20) * random.choice([-1, 1]))
            new_num = max(0, num + change)  # Don't go negative
            new_str = str(new_num)
        
        perturbed = perturbed.replace(original, new_str, 1)
        changes.append(f"Changed number {original} to {new_str}")
        
        return perturbed, changes
    
    def _perturb_entities(self, text: str) -> Optional[Tuple[str, List[str]]]:
        """
        Perturb named entities in the text.
        
        Args:
            text: Text containing named entities
        
        Returns:
            Tuple of (perturbed_text, list of changes) or None if no entities found
        """
        # Common entity swaps
        entity_swaps = {
            'Einstein': ['Newton', 'Tesla', 'Curie', 'Hawking'],
            'Shakespeare': ['Dickens', 'Austen', 'Hemingway', 'Twain'],
            'Napoleon': ['Caesar', 'Alexander', 'Genghis Khan', 'Charlemagne'],
            'Microsoft': ['Apple', 'Google', 'Amazon', 'IBM'],
            'Apple': ['Microsoft', 'Google', 'Samsung', 'Sony'],
            'Google': ['Microsoft', 'Apple', 'Amazon', 'Facebook'],
        }
        
        changes = []
        perturbed = text
        
        for original_entity, alternatives in entity_swaps.items():
            if original_entity in text:
                new_entity = random.choice(alternatives)
                perturbed = text.replace(original_entity, new_entity, 1)
                changes.append(f"Changed entity '{original_entity}' to '{new_entity}'")
                break
        
        if changes:
            return perturbed, changes
        return None
    
    def test_robustness(
        self,
        source: str,
        original: str,
        perturbations: List[Tuple[str, str, List[str]]],
        detection_threshold: Optional[float] = None
    ) -> RobustnessReport:
        """
        Test the robustness of the evaluation system against perturbations.
        
        Args:
            source: Source text for evaluation
            original: Original unperturbed candidate text
            perturbations: List of (perturbed_text, type, changes) tuples
            detection_threshold: Score difference threshold for detection
                               (uses instance default if not provided)
        
        Returns:
            RobustnessReport with detection rates and detailed results
        """
        threshold = detection_threshold or self.detection_threshold
        logger.info(f"Testing robustness with {len(perturbations)} perturbations")
        
        # Evaluate original text
        logger.info("Evaluating original text...")
        original_result = self.toolkit.evaluate(source, original)
        original_score = original_result.consensus_score
        
        # Test each perturbation
        perturbation_results = []
        detected_count = 0
        by_type: Dict[str, Dict[str, Any]] = {}
        
        for perturbed_text, pert_type, changes in perturbations:
            logger.info(f"Testing {pert_type} perturbation...")
            
            # Evaluate perturbed text
            perturbed_result = self.toolkit.evaluate(source, perturbed_text)
            perturbed_score = perturbed_result.consensus_score
            
            # Calculate score delta
            score_delta = original_score - perturbed_score
            
            # Check if perturbation was detected (score should drop)
            detected = score_delta >= threshold
            if detected:
                detected_count += 1
            
            # Record result
            result = PerturbationResult(
                original_text=original,
                perturbed_text=perturbed_text,
                perturbation_type=pert_type,
                perturbations_applied=changes,
                detected=detected,
                original_score=original_score,
                perturbed_score=perturbed_score,
                score_delta=score_delta
            )
            perturbation_results.append(result)
            
            # Update by-type statistics
            if pert_type not in by_type:
                by_type[pert_type] = {
                    'total': 0,
                    'detected': 0,
                    'detection_rate': 0.0
                }
            by_type[pert_type]['total'] += 1
            if detected:
                by_type[pert_type]['detected'] += 1
        
        # Calculate detection rates by type
        for pert_type in by_type:
            total = by_type[pert_type]['total']
            detected = by_type[pert_type]['detected']
            by_type[pert_type]['detection_rate'] = (detected / total * 100) if total > 0 else 0.0
        
        # Calculate overall statistics
        total_tests = len(perturbations)
        missed_count = total_tests - detected_count
        detection_rate = (detected_count / total_tests * 100) if total_tests > 0 else 0.0
        
        # Create report
        report = RobustnessReport(
            total_tests=total_tests,
            detected_count=detected_count,
            missed_count=missed_count,
            detection_rate=detection_rate,
            false_positive_rate=0.0,  # Would need clean examples to calculate
            perturbation_results=perturbation_results,
            by_type=by_type,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'detection_threshold': threshold,
                'original_score': original_score,
                'source_text': source,
            }
        )
        
        logger.info(f"Robustness testing complete: {detection_rate:.1f}% detection rate")
        return report
    
    def test_symmetry(
        self,
        candidate_a: str,
        candidate_b: str,
        source: str
    ) -> SymmetryReport:
        """
        Test pairwise ranking symmetry (A vs B should be consistent with B vs A).
        
        Args:
            candidate_a: First candidate text
            candidate_b: Second candidate text
            source: Source text for comparison
        
        Returns:
            SymmetryReport indicating whether rankings are symmetric
        """
        logger.info("Testing pairwise ranking symmetry...")
        
        # Compare A vs B
        logger.info("Evaluating A vs B...")
        ab_result = self.toolkit.pairwise_compare(source, candidate_a, candidate_b)
        ab_winner = ab_result.get('winner', 'UNKNOWN')
        ab_reasoning = ab_result.get('reasoning', '')
        
        # Compare B vs A
        logger.info("Evaluating B vs A...")
        ba_result = self.toolkit.pairwise_compare(source, candidate_b, candidate_a)
        ba_winner = ba_result.get('winner', 'UNKNOWN')
        ba_reasoning = ba_result.get('reasoning', '')
        
        # Check symmetry
        # If A > B, then B < A (winner should flip)
        # If TIE, both should be TIE
        is_symmetric = False
        if ab_winner == 'TIE' and ba_winner == 'TIE':
            is_symmetric = True
        elif ab_winner == 'A' and ba_winner == 'B':
            is_symmetric = True
        elif ab_winner == 'B' and ba_winner == 'A':
            is_symmetric = True
        
        report = SymmetryReport(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            source=source,
            ab_winner=ab_winner,
            ba_winner=ba_winner,
            is_symmetric=is_symmetric,
            ab_reasoning=ab_reasoning,
            ba_reasoning=ba_reasoning
        )
        
        logger.info(f"Symmetry test complete: {'PASS' if is_symmetric else 'FAIL'}")
        return report

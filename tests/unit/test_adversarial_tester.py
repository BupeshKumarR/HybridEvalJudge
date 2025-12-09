"""
Unit tests for the AdversarialTester component.
"""

import pytest
from unittest.mock import MagicMock, Mock

from llm_judge_auditor.components.adversarial_tester import (
    AdversarialTester,
    PerturbationResult,
    RobustnessReport,
    SymmetryReport,
)
from llm_judge_auditor.models import EvaluationResult


class TestAdversarialTester:
    """Test suite for AdversarialTester class."""
    
    @pytest.fixture
    def mock_toolkit(self):
        """Create a mock evaluation toolkit."""
        toolkit = MagicMock()
        
        # Mock evaluate method
        def mock_evaluate(source, candidate):
            result = MagicMock(spec=EvaluationResult)
            # Return different scores based on content
            if "1889" in candidate:
                result.consensus_score = 95.0
            elif "1890" in candidate or "1888" in candidate:
                result.consensus_score = 70.0  # Lower score for wrong date
            elif "Paris" in candidate:
                result.consensus_score = 95.0
            elif "London" in candidate or "Berlin" in candidate:
                result.consensus_score = 60.0  # Lower score for wrong location
            else:
                result.consensus_score = 85.0
            return result
        
        toolkit.evaluate = Mock(side_effect=mock_evaluate)
        
        # Mock pairwise_compare method
        def mock_pairwise_compare(source, candidate_a, candidate_b):
            # Simple logic: higher score wins
            score_a = 95.0 if "correct" in candidate_a else 70.0
            score_b = 95.0 if "correct" in candidate_b else 70.0
            
            if score_a > score_b:
                winner = 'A'
            elif score_b > score_a:
                winner = 'B'
            else:
                winner = 'TIE'
            
            return {
                'winner': winner,
                'reasoning': f'Candidate {winner} is more accurate'
            }
        
        toolkit.pairwise_compare = Mock(side_effect=mock_pairwise_compare)
        
        return toolkit
    
    @pytest.fixture
    def tester(self, mock_toolkit):
        """Create an AdversarialTester instance."""
        return AdversarialTester(mock_toolkit, detection_threshold=10.0)
    
    def test_initialization(self, mock_toolkit):
        """Test AdversarialTester initialization."""
        tester = AdversarialTester(mock_toolkit, detection_threshold=15.0)
        assert tester.toolkit == mock_toolkit
        assert tester.detection_threshold == 15.0
    
    def test_generate_perturbations_date_shift(self, tester):
        """Test date perturbation generation."""
        text = "The Eiffel Tower was completed in 1889."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["date_shift"],
            num_variants=1
        )
        
        assert len(perturbations) == 1
        perturbed_text, pert_type, changes = perturbations[0]
        assert pert_type == "date_shift"
        assert "1889" not in perturbed_text  # Date should be changed
        assert len(changes) > 0
        assert "year" in changes[0].lower() or "date" in changes[0].lower()
    
    def test_generate_perturbations_location_swap(self, tester):
        """Test location perturbation generation."""
        text = "The Eiffel Tower is located in Paris, France."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["location_swap"],
            num_variants=1
        )
        
        assert len(perturbations) == 1
        perturbed_text, pert_type, changes = perturbations[0]
        assert pert_type == "location_swap"
        # Either Paris or France should be changed
        assert ("Paris" not in perturbed_text) or ("France" not in perturbed_text)
        assert len(changes) > 0
        assert "location" in changes[0].lower()
    
    def test_generate_perturbations_number_change(self, tester):
        """Test number perturbation generation."""
        text = "The tower is 324 meters tall and weighs 10100 tons."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["number_change"],
            num_variants=1
        )
        
        assert len(perturbations) == 1
        perturbed_text, pert_type, changes = perturbations[0]
        assert pert_type == "number_change"
        # At least one number should be changed
        assert perturbed_text != text
        assert len(changes) > 0
        assert "number" in changes[0].lower()
    
    def test_generate_perturbations_entity_replace(self, tester):
        """Test entity perturbation generation."""
        text = "Einstein developed the theory of relativity."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["entity_replace"],
            num_variants=1
        )
        
        assert len(perturbations) == 1
        perturbed_text, pert_type, changes = perturbations[0]
        assert pert_type == "entity_replace"
        assert "Einstein" not in perturbed_text
        assert len(changes) > 0
        assert "entity" in changes[0].lower()
    
    def test_generate_perturbations_multiple_types(self, tester):
        """Test generating multiple perturbation types."""
        text = "The Eiffel Tower was completed in 1889 in Paris."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["date_shift", "location_swap"],
            num_variants=1
        )
        
        assert len(perturbations) == 2
        types = [p[1] for p in perturbations]
        assert "date_shift" in types
        assert "location_swap" in types
    
    def test_generate_perturbations_multiple_variants(self, tester):
        """Test generating multiple variants per type."""
        text = "The Eiffel Tower was completed in 1889."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["date_shift"],
            num_variants=3
        )
        
        assert len(perturbations) == 3
        assert all(p[1] == "date_shift" for p in perturbations)
    
    def test_generate_perturbations_unknown_type(self, tester):
        """Test handling of unknown perturbation type."""
        text = "Some text."
        perturbations = tester.generate_perturbations(
            text=text,
            perturbation_types=["unknown_type"],
            num_variants=1
        )
        
        # Should skip unknown types
        assert len(perturbations) == 0
    
    def test_test_robustness_detection(self, tester, mock_toolkit):
        """Test robustness testing with detected perturbations."""
        source = "The Eiffel Tower was completed in 1889."
        original = "The Eiffel Tower was completed in 1889."
        
        # Create perturbations that should be detected
        perturbations = [
            ("The Eiffel Tower was completed in 1890.", "date_shift", ["Changed year 1889 to 1890"]),
            ("The Eiffel Tower was completed in 1888.", "date_shift", ["Changed year 1889 to 1888"]),
        ]
        
        report = tester.test_robustness(source, original, perturbations)
        
        assert isinstance(report, RobustnessReport)
        assert report.total_tests == 2
        assert report.detected_count == 2  # Both should be detected (score drops by 25)
        assert report.missed_count == 0
        assert report.detection_rate == 100.0
        assert len(report.perturbation_results) == 2
        assert "date_shift" in report.by_type
        assert report.by_type["date_shift"]["total"] == 2
        assert report.by_type["date_shift"]["detected"] == 2
    
    def test_test_robustness_partial_detection(self, tester, mock_toolkit):
        """Test robustness testing with some perturbations missed."""
        source = "The Eiffel Tower was completed in 1889 in Paris."
        original = "The Eiffel Tower was completed in 1889 in Paris."
        
        # Mix of detected and undetected perturbations
        perturbations = [
            ("The Eiffel Tower was completed in 1890 in Paris.", "date_shift", ["Changed year"]),
            ("The Eiffel Tower was completed in 1889 in London.", "location_swap", ["Changed location"]),
        ]
        
        report = tester.test_robustness(source, original, perturbations)
        
        assert report.total_tests == 2
        # At least one should be detected (date shift drops score by 25)
        assert report.detected_count >= 1
        assert 0 <= report.detection_rate <= 100
    
    def test_test_robustness_custom_threshold(self, tester, mock_toolkit):
        """Test robustness testing with custom detection threshold."""
        source = "The Eiffel Tower was completed in 1889."
        original = "The Eiffel Tower was completed in 1889."
        
        perturbations = [
            ("The Eiffel Tower was completed in 1890.", "date_shift", ["Changed year"]),
        ]
        
        # Use a very high threshold - should not detect
        report = tester.test_robustness(
            source, original, perturbations, detection_threshold=50.0
        )
        
        assert report.total_tests == 1
        assert report.detected_count == 0  # Threshold too high
        assert report.detection_rate == 0.0
    
    def test_test_robustness_metadata(self, tester, mock_toolkit):
        """Test that robustness report includes proper metadata."""
        source = "The Eiffel Tower was completed in 1889."
        original = "The Eiffel Tower was completed in 1889."
        
        perturbations = [
            ("The Eiffel Tower was completed in 1890.", "date_shift", ["Changed year"]),
        ]
        
        report = tester.test_robustness(source, original, perturbations)
        
        assert "timestamp" in report.metadata
        assert "detection_threshold" in report.metadata
        assert "original_score" in report.metadata
        assert "source_text" in report.metadata
        assert report.metadata["detection_threshold"] == 10.0
        assert report.metadata["source_text"] == source
    
    def test_test_symmetry_consistent(self, tester, mock_toolkit):
        """Test symmetry testing with consistent rankings."""
        source = "Some source text."
        candidate_a = "This is correct."
        candidate_b = "This is wrong."
        
        report = tester.test_symmetry(candidate_a, candidate_b, source)
        
        assert isinstance(report, SymmetryReport)
        assert report.candidate_a == candidate_a
        assert report.candidate_b == candidate_b
        assert report.source == source
        assert report.is_symmetric  # Should be symmetric based on mock
        assert report.ab_winner in ['A', 'B', 'TIE']
        assert report.ba_winner in ['A', 'B', 'TIE']
    
    def test_test_symmetry_tie(self, tester, mock_toolkit):
        """Test symmetry testing with tie results."""
        source = "Some source text."
        candidate_a = "Some text."
        candidate_b = "Some other text."
        
        report = tester.test_symmetry(candidate_a, candidate_b, source)
        
        assert isinstance(report, SymmetryReport)
        # Based on our mock, both should have same score -> TIE
        if report.ab_winner == 'TIE':
            assert report.ba_winner == 'TIE'
            assert report.is_symmetric
    
    def test_perturbation_result_dataclass(self):
        """Test PerturbationResult dataclass."""
        result = PerturbationResult(
            original_text="Original",
            perturbed_text="Perturbed",
            perturbation_type="date_shift",
            perturbations_applied=["Changed year"],
            detected=True,
            original_score=95.0,
            perturbed_score=70.0,
            score_delta=25.0
        )
        
        assert result.original_text == "Original"
        assert result.perturbed_text == "Perturbed"
        assert result.perturbation_type == "date_shift"
        assert result.detected is True
        assert result.score_delta == 25.0
    
    def test_robustness_report_dataclass(self):
        """Test RobustnessReport dataclass."""
        report = RobustnessReport(
            total_tests=10,
            detected_count=8,
            missed_count=2,
            detection_rate=80.0,
            false_positive_rate=5.0,
            perturbation_results=[],
            by_type={"date_shift": {"total": 5, "detected": 4}},
            metadata={"timestamp": "2024-01-01"}
        )
        
        assert report.total_tests == 10
        assert report.detected_count == 8
        assert report.detection_rate == 80.0
        assert "date_shift" in report.by_type
    
    def test_symmetry_report_dataclass(self):
        """Test SymmetryReport dataclass."""
        report = SymmetryReport(
            candidate_a="A",
            candidate_b="B",
            source="Source",
            ab_winner="A",
            ba_winner="B",
            is_symmetric=True,
            ab_reasoning="A is better",
            ba_reasoning="B is better from this perspective"
        )
        
        assert report.candidate_a == "A"
        assert report.ab_winner == "A"
        assert report.is_symmetric is True
    
    def test_perturb_dates_no_dates(self, tester):
        """Test date perturbation with text containing no dates."""
        text = "This text has no dates."
        result = tester._perturb_dates(text)
        
        assert result is None
    
    def test_perturb_locations_no_locations(self, tester):
        """Test location perturbation with text containing no known locations."""
        text = "This text has no known locations."
        result = tester._perturb_locations(text)
        
        assert result is None
    
    def test_perturb_numbers_no_numbers(self, tester):
        """Test number perturbation with text containing no numbers."""
        text = "This text has no numbers."
        result = tester._perturb_numbers(text)
        
        assert result is None
    
    def test_perturb_entities_no_entities(self, tester):
        """Test entity perturbation with text containing no known entities."""
        text = "This text has no known entities."
        result = tester._perturb_entities(text)
        
        assert result is None
    
    def test_perturb_dates_various_formats(self, tester):
        """Test date perturbation with various date formats."""
        texts = [
            "The event happened in 1989.",
            "It occurred on January 15, 2020.",
            "The date was 12/31/2019.",
        ]
        
        for text in texts:
            result = tester._perturb_dates(text)
            if result:  # Some formats might not be supported
                perturbed, changes = result
                assert perturbed != text
                assert len(changes) > 0
    
    def test_perturb_numbers_filters_years(self, tester):
        """Test that number perturbation filters out years."""
        text = "The year 2020 had 366 days."
        result = tester._perturb_numbers(text)
        
        if result:
            perturbed, changes = result
            # Should change 366, not 2020
            assert "2020" in perturbed
            assert "366" not in perturbed or perturbed != text

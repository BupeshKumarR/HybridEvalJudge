"""
Example usage of the AdversarialTester component.

This script demonstrates how to use the AdversarialTester to test the
robustness of the evaluation toolkit against adversarial perturbations.
"""

from llm_judge_auditor import EvaluationToolkit
from llm_judge_auditor.components import AdversarialTester


def main():
    """Demonstrate adversarial testing functionality."""
    
    print("=" * 80)
    print("Adversarial Testing Example")
    print("=" * 80)
    print()
    
    # Initialize the evaluation toolkit
    print("Initializing evaluation toolkit...")
    try:
        toolkit = EvaluationToolkit.from_preset("fast")
        print("✓ Toolkit initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize toolkit: {e}")
        print("\nNote: This example requires models to be available.")
        print("Run with actual models or use mock for demonstration.")
        return
    
    print()
    
    # Initialize the adversarial tester
    print("Initializing adversarial tester...")
    tester = AdversarialTester(toolkit, detection_threshold=10.0)
    print("✓ Adversarial tester initialized")
    print()
    
    # Example 1: Test date perturbations
    print("-" * 80)
    print("Example 1: Date Perturbation Testing")
    print("-" * 80)
    
    source = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
    original = "The Eiffel Tower was built in 1889 in Paris, France."
    
    print(f"Source: {source}")
    print(f"Original: {original}")
    print()
    
    # Generate date perturbations
    print("Generating date perturbations...")
    date_perturbations = tester.generate_perturbations(
        text=original,
        perturbation_types=["date_shift"],
        num_variants=3
    )
    
    print(f"Generated {len(date_perturbations)} date perturbations:")
    for i, (perturbed, pert_type, changes) in enumerate(date_perturbations, 1):
        print(f"  {i}. {perturbed}")
        print(f"     Changes: {', '.join(changes)}")
    print()
    
    # Test robustness
    print("Testing robustness against date perturbations...")
    date_report = tester.test_robustness(
        source=source,
        original=original,
        perturbations=date_perturbations
    )
    
    print(f"Detection Rate: {date_report.detection_rate:.1f}%")
    print(f"Detected: {date_report.detected_count}/{date_report.total_tests}")
    print()
    
    # Example 2: Test location perturbations
    print("-" * 80)
    print("Example 2: Location Perturbation Testing")
    print("-" * 80)
    
    print(f"Source: {source}")
    print(f"Original: {original}")
    print()
    
    # Generate location perturbations
    print("Generating location perturbations...")
    location_perturbations = tester.generate_perturbations(
        text=original,
        perturbation_types=["location_swap"],
        num_variants=2
    )
    
    print(f"Generated {len(location_perturbations)} location perturbations:")
    for i, (perturbed, pert_type, changes) in enumerate(location_perturbations, 1):
        print(f"  {i}. {perturbed}")
        print(f"     Changes: {', '.join(changes)}")
    print()
    
    # Test robustness
    print("Testing robustness against location perturbations...")
    location_report = tester.test_robustness(
        source=source,
        original=original,
        perturbations=location_perturbations
    )
    
    print(f"Detection Rate: {location_report.detection_rate:.1f}%")
    print(f"Detected: {location_report.detected_count}/{location_report.total_tests}")
    print()
    
    # Example 3: Test multiple perturbation types
    print("-" * 80)
    print("Example 3: Mixed Perturbation Testing")
    print("-" * 80)
    
    text_with_numbers = "The tower is 324 meters tall and was completed in 1889."
    source_with_numbers = "The Eiffel Tower stands 324 meters tall and was finished in 1889."
    
    print(f"Source: {source_with_numbers}")
    print(f"Original: {text_with_numbers}")
    print()
    
    # Generate multiple types of perturbations
    print("Generating mixed perturbations...")
    mixed_perturbations = tester.generate_perturbations(
        text=text_with_numbers,
        perturbation_types=["date_shift", "number_change"],
        num_variants=2
    )
    
    print(f"Generated {len(mixed_perturbations)} mixed perturbations:")
    for i, (perturbed, pert_type, changes) in enumerate(mixed_perturbations, 1):
        print(f"  {i}. [{pert_type}] {perturbed}")
        print(f"     Changes: {', '.join(changes)}")
    print()
    
    # Test robustness
    print("Testing robustness against mixed perturbations...")
    mixed_report = tester.test_robustness(
        source=source_with_numbers,
        original=text_with_numbers,
        perturbations=mixed_perturbations
    )
    
    print(f"Overall Detection Rate: {mixed_report.detection_rate:.1f}%")
    print(f"Detected: {mixed_report.detected_count}/{mixed_report.total_tests}")
    print()
    
    print("Detection by type:")
    for pert_type, stats in mixed_report.by_type.items():
        print(f"  {pert_type}: {stats['detection_rate']:.1f}% "
              f"({stats['detected']}/{stats['total']})")
    print()
    
    # Example 4: Test pairwise ranking symmetry
    print("-" * 80)
    print("Example 4: Pairwise Ranking Symmetry Testing")
    print("-" * 80)
    
    candidate_a = "The Eiffel Tower was completed in 1889 in Paris."
    candidate_b = "The Eiffel Tower was finished in 1890 in Paris."
    
    print(f"Source: {source}")
    print(f"Candidate A: {candidate_a}")
    print(f"Candidate B: {candidate_b}")
    print()
    
    print("Testing pairwise ranking symmetry...")
    symmetry_report = tester.test_symmetry(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        source=source
    )
    
    print(f"A vs B Winner: {symmetry_report.ab_winner}")
    print(f"B vs A Winner: {symmetry_report.ba_winner}")
    print(f"Is Symmetric: {'✓ Yes' if symmetry_report.is_symmetric else '✗ No'}")
    print()
    
    if symmetry_report.ab_reasoning:
        print(f"A vs B Reasoning: {symmetry_report.ab_reasoning}")
    if symmetry_report.ba_reasoning:
        print(f"B vs A Reasoning: {symmetry_report.ba_reasoning}")
    print()
    
    # Example 5: Detailed perturbation results
    print("-" * 80)
    print("Example 5: Detailed Perturbation Analysis")
    print("-" * 80)
    
    print("Analyzing individual perturbation results...")
    print()
    
    for i, result in enumerate(mixed_report.perturbation_results[:3], 1):
        print(f"Perturbation {i}:")
        print(f"  Type: {result.perturbation_type}")
        print(f"  Original Score: {result.original_score:.1f}")
        print(f"  Perturbed Score: {result.perturbed_score:.1f}")
        print(f"  Score Delta: {result.score_delta:.1f}")
        print(f"  Detected: {'✓ Yes' if result.detected else '✗ No'}")
        print(f"  Changes: {', '.join(result.perturbations_applied)}")
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Adversarial testing helps evaluate the robustness of the evaluation")
    print("toolkit by testing its ability to detect subtle factual perturbations.")
    print()
    print("Key metrics:")
    print(f"  - Date perturbations: {date_report.detection_rate:.1f}% detected")
    print(f"  - Location perturbations: {location_report.detection_rate:.1f}% detected")
    print(f"  - Mixed perturbations: {mixed_report.detection_rate:.1f}% detected")
    print(f"  - Pairwise symmetry: {'Consistent' if symmetry_report.is_symmetric else 'Inconsistent'}")
    print()


if __name__ == "__main__":
    main()

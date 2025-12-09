"""
Example usage of the ClaimRouter component.

This script demonstrates how to use the ClaimRouter to classify claims
and route them to specialized judges based on their expertise areas.
"""

from llm_judge_auditor.components.claim_router import ClaimRouter
from llm_judge_auditor.models import Claim, ClaimType


def main():
    """Demonstrate ClaimRouter functionality."""
    
    print("=" * 70)
    print("ClaimRouter Example - Specialized Judge Selection")
    print("=" * 70)
    print()
    
    # Example 1: Basic claim classification
    print("Example 1: Claim Classification")
    print("-" * 70)
    
    router = ClaimRouter()
    
    test_claims = [
        "The unemployment rate increased by 5.2% last quarter.",
        "The Eiffel Tower was completed in 1889.",
        "If the temperature rises, then the ice will melt.",
        "People generally feel happier when the weather is warm.",
        "Paris is the capital of France.",
    ]
    
    for claim_text in test_claims:
        claim = Claim(text=claim_text, source_span=(0, len(claim_text)))
        claim_type = router.classify_claim(claim)
        print(f"Claim: {claim_text}")
        print(f"Type:  {claim_type.value}")
        print()
    
    # Example 2: Routing claims to specialized judges
    print("\nExample 2: Routing to Specialized Judges")
    print("-" * 70)
    
    # Define judge specializations
    specializations = {
        "llama-3-8b": ["factual", "logical"],
        "mistral-7b": ["numerical", "temporal"],
        "phi-3-mini": ["commonsense"],
    }
    
    router = ClaimRouter(specializations)
    
    # Create claims
    claims = [
        Claim(text="The temperature was 25 degrees Celsius.", source_span=(0, 40)),
        Claim(text="Paris is the capital of France.", source_span=(0, 32)),
        Claim(text="The conference was held in September 2023.", source_span=(0, 42)),
        Claim(text="People usually prefer warm weather to cold weather.", source_span=(0, 52)),
        Claim(text="If it rains, then the ground will be wet.", source_span=(0, 41)),
    ]
    
    available_judges = ["llama-3-8b", "mistral-7b", "phi-3-mini"]
    
    print(f"Available judges: {', '.join(available_judges)}")
    print()
    
    for claim in claims:
        selected_judge = router.route_to_judge(claim, available_judges)
        print(f"Claim: {claim.text}")
        print(f"Type:  {claim.claim_type.value}")
        print(f"Judge: {selected_judge}")
        print()
    
    # Example 3: Batch routing
    print("\nExample 3: Batch Routing")
    print("-" * 70)
    
    routing = router.route_claims_to_judges(claims, available_judges)
    
    print("Claims grouped by assigned judge:")
    print()
    for judge_name, judge_claims in routing.items():
        if judge_claims:
            print(f"{judge_name}:")
            for claim in judge_claims:
                print(f"  - {claim.text[:60]}...")
            print()
    
    # Example 4: Managing specializations
    print("\nExample 4: Managing Specializations")
    print("-" * 70)
    
    # Get specializations for a judge
    specs = router.get_judge_specializations("mistral-7b")
    print(f"Mistral-7B specializations: {specs}")
    
    # Get judges specialized in a claim type
    numerical_judges = router.get_specialized_judges(ClaimType.NUMERICAL)
    print(f"Judges specialized in numerical claims: {numerical_judges}")
    
    # Update specializations dynamically
    router.update_specializations("llama-3-8b", ["factual", "logical", "numerical"])
    print(f"\nUpdated Llama-3-8B specializations: {router.get_judge_specializations('llama-3-8b')}")
    
    numerical_judges = router.get_specialized_judges(ClaimType.NUMERICAL)
    print(f"Judges specialized in numerical claims (after update): {numerical_judges}")
    
    # Example 5: Routing without specializations (general-purpose judges)
    print("\n\nExample 5: General-Purpose Judges (No Specializations)")
    print("-" * 70)
    
    general_router = ClaimRouter()  # No specializations
    
    claim = Claim(text="The GDP grew by 3.5% in Q4.", source_span=(0, 28))
    selected = general_router.route_to_judge(claim, ["judge-a", "judge-b", "judge-c"])
    
    print(f"Claim: {claim.text}")
    print(f"Type:  {claim.claim_type.value}")
    print(f"Selected judge: {selected}")
    print("(First available judge used when no specializations defined)")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

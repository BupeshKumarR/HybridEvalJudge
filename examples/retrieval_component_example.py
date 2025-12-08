"""
Example usage of the RetrievalComponent.

This script demonstrates:
1. Initializing the retrieval component
2. Loading a knowledge base
3. Extracting claims from text
4. Retrieving relevant passages
5. Operating in fallback mode
"""

import tempfile
from pathlib import Path

from llm_judge_auditor.components.retrieval_component import RetrievalComponent


def create_sample_knowledge_base():
    """Create a sample knowledge base file for demonstration."""
    kb_content = """Wikipedia:Eiffel_Tower\tThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
Wikipedia:Eiffel_Tower\tIt is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Wikipedia:Eiffel_Tower\tConstructed from 1887 to 1889, it was initially criticized by some of France's leading artists and intellectuals.
Wikipedia:Eiffel_Tower\tThe tower is 330 metres tall, about the same height as an 81-storey building.
Wikipedia:Paris\tParis is the capital and most populous city of France.
Wikipedia:Paris\tSince the 17th century, Paris has been one of the world's major centres of finance, diplomacy, commerce, fashion, gastronomy, and science.
Wikipedia:Gustave_Eiffel\tAlexandre Gustave Eiffel was a French civil engineer and architect.
Wikipedia:Gustave_Eiffel\tHe is best known for the world-famous Eiffel Tower, built for the 1889 Universal Exposition in Paris.
Wikipedia:1889_World_Fair\tThe Exposition Universelle of 1889 was a world's fair held in Paris from 6 May to 31 October 1889.
Wikipedia:1889_World_Fair\tIt was held to celebrate the 100th anniversary of the storming of the Bastille."""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(kb_content)
        return f.name


def main():
    print("=" * 80)
    print("RetrievalComponent Example")
    print("=" * 80)
    
    # Example 1: Basic initialization
    print("\n1. Initializing RetrievalComponent...")
    component = RetrievalComponent(
        embedding_model="all-MiniLM-L6-v2",
        top_k=3,
        device="cpu",
    )
    print(f"   Embedding model: {component.embedding_model_name}")
    print(f"   Top-k: {component.top_k}")
    print(f"   Device: {component.device}")
    print(f"   Fallback mode: {component.fallback_mode()}")
    
    # Example 2: Extract claims from text
    print("\n2. Extracting claims from text...")
    candidate_text = """The Eiffel Tower was completed in 1889 for the World's Fair. 
It is located in Paris, France. The tower was designed by Gustave Eiffel. 
It stands 330 meters tall and is one of the most visited monuments in the world."""
    
    claims = component.extract_claims(candidate_text)
    print(f"   Extracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"   {i}. [{claim.claim_type.value}] {claim.text}")
        print(f"      Position: {claim.source_span}")
    
    # Example 3: Operating in fallback mode (no KB)
    print("\n3. Attempting retrieval in fallback mode...")
    if claims:
        passages = component.retrieve_passages(claims[0])
        print(f"   Retrieved {len(passages)} passages (expected 0 in fallback mode)")
    
    # Example 4: Initialize knowledge base
    print("\n4. Initializing knowledge base...")
    kb_path = create_sample_knowledge_base()
    try:
        component.initialize_knowledge_base(kb_path)
        print(f"   Knowledge base loaded successfully")
        print(f"   Fallback mode: {component.fallback_mode()}")
        
        stats = component.get_stats()
        print(f"   Number of passages: {stats['num_passages']}")
        print(f"   Index size: {stats['index_size']}")
        
        # Example 5: Retrieve passages for claims
        print("\n5. Retrieving passages for claims...")
        for i, claim in enumerate(claims[:2], 1):  # Just first 2 claims
            print(f"\n   Claim {i}: {claim.text}")
            passages = component.retrieve_passages(claim)
            print(f"   Retrieved {len(passages)} passages:")
            
            for j, passage in enumerate(passages, 1):
                print(f"      {j}. [Score: {passage.relevance_score:.3f}] {passage.source}")
                print(f"         {passage.text[:100]}...")
        
        # Example 6: Custom top_k
        print("\n6. Retrieving with custom top_k...")
        if claims:
            passages = component.retrieve_passages(claims[0], top_k=5)
            print(f"   Retrieved {len(passages)} passages with top_k=5")
        
        # Example 7: Statistics
        print("\n7. Component statistics...")
        stats = component.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    finally:
        # Clean up temporary file
        Path(kb_path).unlink()
        print("\n8. Cleaned up temporary knowledge base")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

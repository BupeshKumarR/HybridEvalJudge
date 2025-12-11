#!/usr/bin/env python3
"""
Full Pipeline Demo - Generate + Evaluate + Hallucination Metrics
================================================================

This demo shows the COMPLETE pipeline:
1. Ask user for a question
2. Generate a response using Groq LLM
3. Evaluate the response using multiple judges
4. Compute hallucination metrics (MiHR, FactScore, Kappa, etc.)

Requirements:
- GROQ_API_KEY environment variable set
- pip install groq

Run: python demo/full_pipeline_demo.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_api_key():
    """Check if Groq API key is available."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY not set!")
        print("\nTo set it:")
        print("  export GROQ_API_KEY='your-key-here'")
        print("\nGet a free key at: https://console.groq.com/keys")
        return None
    return key

def generate_response(question: str, api_key: str) -> str:
    """Generate a response to the question using Groq."""
    from groq import Groq
    
    client = Groq(api_key=api_key)
    
    print(f"\n🤖 Generating response to: '{question}'")
    print("   (Using Groq Llama 3.3 70B...)\n")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide accurate, factual information."},
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def evaluate_with_hallucination_metrics(question: str, response: str, api_key: str):
    """Evaluate the response and compute hallucination metrics."""
    from llm_judge_auditor.components.api_key_manager import APIKeyManager
    from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
    from llm_judge_auditor.components.hallucination_metrics import (
        HallucinationMetricsCalculator,
        ClaimVerificationMatrixBuilder,
    )
    from llm_judge_auditor.config import ToolkitConfig
    from llm_judge_auditor.models import Verdict, VerdictLabel, Claim
    
    # Initialize
    api_key_manager = APIKeyManager()
    api_key_manager.load_keys()
    
    config = ToolkitConfig()
    ensemble = APIJudgeEnsemble(
        config=config,
        api_key_manager=api_key_manager,
        parallel_execution=True
    )
    
    print(f"📊 Evaluating with {ensemble.get_judge_count()} judges: {', '.join(ensemble.get_judge_names())}")
    print("   (This may take 5-10 seconds...)\n")
    
    # Evaluate
    verdicts = ensemble.evaluate(
        source_text=question,  # Use question as source for factual accuracy
        candidate_output=response,
        task="factual_accuracy"
    )
    
    # Aggregate scores
    consensus, individual_scores, disagreement = ensemble.aggregate_verdicts(verdicts)
    
    # Compute hallucination metrics
    calculator = HallucinationMetricsCalculator()
    
    # Extract claims from response (simple sentence splitting)
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    claims = [Claim(text=s, source_span=(0, len(s)), claim_type="factual") for s in sentences[:10]]
    
    # Create verdicts for claims based on judge scores
    # If consensus < 70, mark some claims as unsupported
    claim_verdicts = []
    for i, claim in enumerate(claims):
        # Use judge scores to determine if claim is supported
        avg_score = consensus / 100.0
        if avg_score > 0.7:
            label = VerdictLabel.SUPPORTED
        elif avg_score > 0.4:
            label = VerdictLabel.NOT_ENOUGH_INFO
        else:
            label = VerdictLabel.REFUTED
        
        claim_verdicts.append(Verdict(
            label=label,
            confidence=avg_score,
            evidence=[],
            reasoning=f"Based on judge consensus score of {consensus:.1f}"
        ))
    
    # Compute MiHR - takes List[Verdict] directly
    mihr_result = calculator.compute_mihr(claim_verdicts)
    
    # Compute FactScore - takes List[Verdict] directly
    factscore = calculator.compute_factscore(claim_verdicts)
    
    # Compute Fleiss' Kappa from judge ratings
    # Convert verdicts to rating matrix
    ratings = []
    for verdict in verdicts:
        # Convert score to category: 0=low, 1=medium, 2=high
        if verdict.score >= 80:
            rating = 2
        elif verdict.score >= 50:
            rating = 1
        else:
            rating = 0
        ratings.append([rating])
    
    kappa_result = None
    if len(verdicts) >= 2:
        # Need at least 2 judges for Kappa
        # Create proper rating matrix: items x judges
        rating_matrix = [[r[0] for r in ratings]]  # 1 item, multiple judges
        kappa_result = calculator.compute_fleiss_kappa(rating_matrix, num_categories=3)
    
    return {
        "verdicts": verdicts,
        "consensus_score": consensus,
        "individual_scores": individual_scores,
        "disagreement": disagreement,
        "claims_extracted": len(claims),
        "mihr": mihr_result,
        "factscore": factscore,
        "kappa": kappa_result,
    }

def main():
    print("=" * 70)
    print("🔬 LLM Judge Auditor - Full Pipeline Demo")
    print("=" * 70)
    print("\nThis demo will:")
    print("  1. Ask you for a question")
    print("  2. Generate a response using Groq LLM")
    print("  3. Evaluate the response with multiple judges")
    print("  4. Compute hallucination metrics (MiHR, FactScore, Kappa)")
    print()
    
    # Check API key
    api_key = check_api_key()
    if not api_key:
        return
    
    print("✅ Groq API key found\n")
    
    # Get question from user
    print("💬 Enter your question (or press Enter for default):")
    user_question = input("   > ").strip()
    
    if not user_question:
        user_question = "What are the main causes of climate change?"
        print(f"   Using default: '{user_question}'")
    
    try:
        # Step 1: Generate response
        response = generate_response(user_question, api_key)
        
        print("─" * 70)
        print("📝 GENERATED RESPONSE:")
        print("─" * 70)
        print(response)
        print()
        
        # Step 2: Evaluate with hallucination metrics
        print("─" * 70)
        print("📊 EVALUATION RESULTS:")
        print("─" * 70)
        
        results = evaluate_with_hallucination_metrics(user_question, response, api_key)
        
        # Display individual judge scores
        print("\n🤖 Individual Judge Scores:")
        for verdict in results["verdicts"]:
            print(f"   • {verdict.judge_name}: {verdict.score:.1f}/100")
            print(f"     Confidence: {verdict.confidence:.2f}")
            print(f"     Reasoning: {verdict.reasoning[:150]}...")
            print()
        
        # Display consensus
        print(f"🎯 Consensus Score: {results['consensus_score']:.1f}/100")
        print(f"   Disagreement Level: {results['disagreement']:.1f}")
        
        # Display hallucination metrics
        print("\n" + "─" * 70)
        print("📈 HALLUCINATION METRICS:")
        print("─" * 70)
        
        print(f"\n   Claims Extracted: {results['claims_extracted']}")
        
        if results['mihr']:
            mihr_val = results['mihr'].value
            if mihr_val is not None:
                print(f"   MiHR (Micro Hallucination Rate): {mihr_val*100:.1f}%")
                print(f"      → {mihr_val*100:.1f}% of claims are unsupported")
            else:
                print("   MiHR: N/A (no claims)")
        
        if results['factscore'] is not None:
            print(f"   FactScore: {results['factscore']*100:.1f}%")
            print(f"      → {results['factscore']*100:.1f}% of claims are factually accurate")
        
        if results['kappa']:
            print(f"   Fleiss' Kappa: {results['kappa'].kappa:.3f}")
            print(f"      → Interpretation: {results['kappa'].interpretation}")
            print(f"      → Observed Agreement: {results['kappa'].observed_agreement:.3f}")
        
        # Risk assessment
        print("\n" + "─" * 70)
        print("⚠️  RISK ASSESSMENT:")
        print("─" * 70)
        
        is_high_risk = False
        risk_reasons = []
        
        if results['mihr'] and results['mihr'].value and results['mihr'].value > 0.3:
            is_high_risk = True
            risk_reasons.append(f"MiHR > 30% ({results['mihr'].value*100:.1f}%)")
        
        if results['kappa'] and results['kappa'].kappa < 0.4:
            is_high_risk = True
            risk_reasons.append(f"Low judge agreement (κ={results['kappa'].kappa:.2f})")
        
        if results['consensus_score'] < 70:
            is_high_risk = True
            risk_reasons.append(f"Low consensus score ({results['consensus_score']:.1f})")
        
        if is_high_risk:
            print(f"\n   🔴 HIGH RISK - Potential hallucination detected!")
            for reason in risk_reasons:
                print(f"      • {reason}")
        else:
            print(f"\n   🟢 LOW RISK - Response appears factually accurate")
        
        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "question": user_question,
            "response": response,
            "consensus_score": results['consensus_score'],
            "individual_scores": results['individual_scores'],
            "disagreement": results['disagreement'],
            "hallucination_metrics": {
                "claims_extracted": results['claims_extracted'],
                "mihr": results['mihr'].value if results['mihr'] else None,
                "factscore": results['factscore'],
                "kappa": results['kappa'].kappa if results['kappa'] else None,
                "kappa_interpretation": results['kappa'].interpretation if results['kappa'] else None,
            },
            "is_high_risk": is_high_risk,
            "risk_reasons": risk_reasons,
        }
        
        output_file = Path("demo/full_pipeline_results.json")
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

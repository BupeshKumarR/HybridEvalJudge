# Implementation Plan

- [x] 1. Set up project structure and core infrastructure
  - Create directory structure for models, components, tests, and configuration
  - Set up Python package with pyproject.toml and requirements.txt
  - Configure development tools (pytest, hypothesis)
  - Create base configuration schema using pydantic
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement basic Device Manager for hardware detection
  - Create simple DeviceManager to detect CUDA, MPS, or CPU 
  - Add basic auto-configuration for device selection
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 3. Implement Model Manager for loading and initializing models
  - Create ModelManager class with lazy loading support
  - Implement verifier model loading with HuggingFace Transformers
  - Implement judge ensemble loading (2-3 models)
  - Add 8-bit quantization support using bitsandbytes
  - Implement model readiness verification
  - Add error handling for model load failures with clear error messages
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3.1 Implement Model Download Utility
  - Create ModelDownloader class to download models from HuggingFace Hub
  - Implement SHA256 verification for downloaded models
  - Save models to ~/.cache/llm-judge-auditor/ directory
  - Add disk space checking before download
  - Prevent re-downloading if model already exists and is valid
  - Add progress bars for download status
  - Handle network errors and retry logic
  - _Requirements: 1.1, 1.2_

- [x] 3.2 Write property test for model initialization
  - **Property 1: Model initialization completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

- [x] 4. Implement simple Preset Manager
  - Create PresetManager with 2-3 basic presets (fast, balanced)
  - Implement preset loading from simple config dict
  - _Requirements: 7.1_

- [x] 5. Implement core data models
  - Create Claim, Passage, Issue, Verdict dataclasses
  - Create EvaluationRequest and EvaluationResult dataclasses
  - Create ToolkitConfig with basic validation
  - Implement JSON serialization for results
  - _Requirements: 8.1, 8.2_

- [x] 6. Implement basic Retrieval Component
  - Create RetrievalComponent class with simple FAISS integration
  - Implement basic claim extraction (sentence splitting)
  - Implement passage retrieval with sentence transformers
  - Add zero-retrieval fallback mode
  - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.6, 6.7_

- [x] 6.1 Write property test for retrieval fallback
  - **Property 13: Retrieval fallback behavior**
  - **Validates: Requirements 6.5, 6.6, 6.7**

- [x] 7. Implement Specialized Verifier
  - Create SpecializedVerifier class with statement-level verification
  - Implement three-way classification (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
  - Add confidence scoring
  - Integrate with retrieval component for context
  - _Requirements: 2.1, 2.2_

- [x] 8. Implement simple Prompt Manager
  - Create PromptManager with hardcoded templates for factual accuracy and pairwise comparison
  - Add basic variable substitution
  - Include chain-of-thought instructions in prompts
  - _Requirements: 7.1, 7.2_

- [x] 9. Implement Judge Model Ensemble
  - Create JudgeEnsemble class with multi-model evaluation
  - Implement single judge evaluation with structured prompts
  - Implement ensemble evaluation (sequential processing)
  - Add basic pairwise comparison functionality
  - Extract scores (0-100) and reasoning from judge outputs
  - _Requirements: 2.3, 2.4, 4.1, 4.2, 4.3_

- [x] 9.1 Write property test for score bounds
  - **Property 3: Score bounds validity**
  - **Validates: Requirements 2.4**

- [x] 9.2 Write property test for pairwise symmetry
  - **Property 8: Pairwise ranking symmetry**
  - **Validates: Requirements 10.2**

- [x] 10. Implement simple Aggregation Engine
  - Create AggregationEngine class with mean and weighted average strategies
  - Add basic disagreement detection (variance > 20 points)
  - Combine verifier verdicts with judge scores
  - Report individual scores alongside consensus
  - _Requirements: 2.5, 11.1, 11.2, 11.3, 11.4_

- [x] 10.1 Write property test for aggregation correctness
  - **Property 24: Ensemble aggregation correctness**
  - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**

- [x] 11. Implement basic bias detection functionality
  - Add bias detection prompt construction
  - Implement phrase flagging with explanations
  - Add severity rating (low, medium, high)
  - _Requirements: 3.1, 3.2, 3.4_

- [x] 12. Implement main Evaluation Toolkit orchestrator
  - Create EvaluationToolkit class as main entry point
  - Implement multi-stage pipeline: retrieval → verifier → ensemble → aggregation
  - Add single evaluation method
  - Integrate all components (retrieval, verifier, judges, aggregation)
  - Ensure correct execution order and data flow
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 12.1 Write property test for pipeline correctness
  - **Property 2: Multi-stage pipeline correctness**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement basic batch processing
  - Add batch evaluation method to EvaluationToolkit
  - Implement sequential processing of requests
  - Add error resilience (continue on failure)
  - Implement basic batch statistics (mean, median)
  - Save batch results to JSON
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 15. Implement simple Report Generator
  - Create ReportGenerator class with basic reporting
  - Include metadata (timestamp, model versions)
  - Include chain-of-thought reasoning from each judge
  - Add confidence levels and disagreement metrics
  - List individual verdicts alongside consensus
  - Implement JSON export and simple Markdown format
  - _Requirements: 8.1, 8.2, 8.4, 8.5, 8.6_

- [x] 16. Implement basic error handling
  - Add malformed output parsing with partial results
  - Implement timeout handling
  - Add basic error logging
  - _Requirements: 9.1, 9.2_

- [x] 17. Create configuration files and templates
  - Create simple config.yaml with essential options
  - Create 2 preset configurations (fast, balanced)
  - Create prompt templates for factual accuracy and pairwise comparison
  - _Requirements: 7.1, 7.2_

- [x] 18. Create simple command-line interface
  - Implement basic CLI with argparse
  - Add commands: evaluate, batch-evaluate
  - Support config file and preset selection
  - _Requirements: 1.1, 2.1, 5.1_

- [x] 19. Create example scripts and documentation
  - Write basic evaluation example
  - Write batch processing example
  - Create README with installation and usage instructions
  - _Requirements: All_

- [x] 20. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. Basic integration testing
  - Test full pipeline end-to-end with all components
  - Test both presets (fast, balanced)
  - Test error handling scenarios
  - Verify core property tests pass with 100+ iterations
  - _Requirements: All_

- [x] 22. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Optional Tasks (Phase 2 / Future Enhancements)

- [x] 23. Implement Streaming Evaluator for large documents
  - Create StreamingEvaluator class with chunking support
  - Implement incremental processing of document streams
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 24. Implement Plugin System
  - Create PluginRegistry class with registration methods
  - Implement plugin discovery from plugins/ directory
  - Support custom verifiers, judges, and aggregators
  - _Requirements: 7.2_

- [x] 25. Implement Adversarial Test Harness
  - Create AdversarialTester class with perturbation generation
  - Implement perturbation types (date_shift, location_swap, number_change)
  - Generate robustness reports with detection rates
  - _Requirements: 14.1, 14.2, 14.5_

- [x] 26. Implement reliability validation features
  - Add evaluation consistency checking (variance < 5 points)
  - Implement Cohen's kappa calculation for inter-model agreement
  - Add Kendall's Tau and Spearman correlation for ranking validation
  - _Requirements: 10.1, 10.4, 10.5_

- [x] 27. Implement component performance tracking
  - Add separate metrics tracking for verifier and judge ensemble
  - Track accuracy, latency, and confidence for each component
  - Log disagreements between verifier and judges
  - _Requirements: 13.1, 13.2, 13.3_

- [x] 28. Implement fine-tuning support for specialized verifiers
  - Create VerifierTrainer class for fine-tuning small models
  - Support FEVER and custom training data formats
  - Implement binary/ternary classification training
  - _Requirements: 12.1, 12.2, 12.3_

- [x] 29. Set up benchmark validation
  - Download FEVER dataset
  - Download TruthfulQA dataset
  - Implement benchmark evaluation scripts
  - Run validation and compare to baselines
  - _Requirements: 10.3_

- [x] 30. Advanced reporting features
  - Add CSV export format
  - Implement retrieval provenance tracking
  - Categorize hallucinations by type
  - _Requirements: 8.3, 8.7, 8.8_

- [x] 31. Implement Claim Router for specialized judge selection
  - Create ClaimRouter class with claim type classification
  - Implement routing logic to match claims to specialized judges
  - _Requirements: 2.3_

- [x] 32. Performance optimization
  - Profile code to identify bottlenecks
  - Optimize model loading and caching
  - Add parallel processing for judge ensemble
  - _Requirements: 1.1, 1.2, 5.1_

---

## Hallucination Quantification Tasks (Requirements 15-20)

- [x] 33. Implement Hallucination Metrics Calculator core
  - [x] 33.1 Create HallucinationMetricsCalculator class with data structures
    - Create ConsensusF1Result, KappaResult, UncertaintyResult, HallucinationProfile dataclasses
    - Add configuration for thresholds (MiHR > 0.3, Kappa < 0.4, uncertainty > 0.8)
    - _Requirements: 15.1, 15.2, 19.1_
  - [x] 33.2 Implement MiHR computation
    - Compute MiHR = unsupported_claims / total_claims
    - Handle zero claims edge case (return None with flag)
    - Ensure output in range [0.0, 1.0]
    - _Requirements: 15.1, 15.3, 15.4, 15.5_
  - [x] 33.3 Implement MaHR computation
    - Compute MaHR = responses_with_hallucinations / total_responses
    - Ensure output in range [0.0, 1.0]
    - _Requirements: 15.2, 15.4_
  - [x] 33.4 Write property test for MiHR/MaHR
    - **Property 31: MiHR and MaHR computation correctness**
    - **Validates: Requirements 15.1, 15.2, 15.4**

- [x] 34. Implement FactScore and Consensus F1
  - [x] 34.1 Implement FactScore computation
    - Compute FactScore = verified_claims / total_claims
    - Ensure output in range [0.0, 1.0]
    - _Requirements: 16.1_
  - [x] 34.2 Implement ClaimVerificationMatrixBuilder
    - Build matrix tracking which claims appear in which model responses
    - Matrix dimensions: (num_unique_claims × num_models)
    - _Requirements: 16.2_
  - [x] 34.3 Implement Consensus F1 computation
    - Compute precision = model_claims_supported_by_others / model_claims
    - Compute recall = consensus_claims_included / total_consensus_claims
    - Compute F1 = 2 × (precision × recall) / (precision + recall)
    - Handle zero division (return 0.0)
    - _Requirements: 16.3, 16.4, 16.5_
  - [x] 34.4 Write property test for FactScore and Consensus F1
    - **Property 32: FactScore and Consensus F1 formula correctness**
    - **Validates: Requirements 16.1, 16.3, 16.4, 16.5**
  - [x] 34.5 Write property test for claim verification matrix
    - **Property 33: Claim verification matrix construction**
    - **Validates: Requirements 16.2**

- [x] 35. Implement Fleiss' Kappa for inter-judge agreement
  - [x] 35.1 Implement Fleiss' Kappa computation
    - Compute observed agreement (Po)
    - Compute expected agreement (Pe)
    - Compute κ = (Po - Pe) / (1 - Pe)
    - Handle fewer than 2 judges (return undefined with error)
    - _Requirements: 17.1, 17.2, 17.4_
  - [x] 35.2 Implement Kappa interpretation
    - Map kappa values to labels: poor (<0.2), fair (0.2-0.4), moderate (0.4-0.6), substantial (0.6-0.8), almost perfect (>0.8)
    - _Requirements: 17.3_
  - [x] 35.3 Write property test for Fleiss' Kappa
    - **Property 34: Fleiss' Kappa computation correctness**
    - **Validates: Requirements 17.1, 17.2, 17.3**

- [x] 36. Implement uncertainty quantification
  - [x] 36.1 Implement Shannon entropy computation
    - Compute H(p) = -Σ pᵢ log pᵢ for probability distributions
    - Handle edge cases (zero probabilities)
    - _Requirements: 18.1_
  - [x] 36.2 Implement epistemic/aleatoric decomposition
    - Compute epistemic = Var(E[p]) across inference samples
    - Compute aleatoric = E[Var(p)] within inference samples
    - Compute total = epistemic + aleatoric
    - _Requirements: 18.2, 18.3, 18.5_
  - [x] 36.3 Implement high uncertainty flagging
    - Flag responses where uncertainty exceeds threshold
    - _Requirements: 18.4_
  - [x] 36.4 Write property test for uncertainty quantification
    - **Property 35: Uncertainty quantification correctness**
    - **Validates: Requirements 18.1, 18.2, 18.3, 18.5**
  - [x] 36.5 Write property test for high uncertainty flagging
    - **Property 36: High uncertainty flagging**
    - **Validates: Requirements 18.4**

- [x] 37. Implement Hallucination Profile generation
  - [x] 37.1 Implement profile compilation
    - Compile MiHR, MaHR, FactScore, F1, Kappa, uncertainty into profile
    - Include claim-level analysis (disputed and consensus claims)
    - _Requirements: 19.1, 19.3_
  - [x] 37.2 Implement reliability classification
    - Assign high/medium/low based on thresholds
    - Flag high risk when MiHR > 0.3 or Kappa < 0.4 or uncertainty > 0.8
    - _Requirements: 19.2, 19.5_
  - [x] 37.3 Implement JSON serialization
    - Serialize profile to JSON format
    - Ensure round-trip consistency
    - _Requirements: 19.4_
  - [x] 37.4 Write property test for hallucination profile
    - **Property 37: Hallucination profile completeness and serialization**
    - **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
  - [x] 37.5 Write property test for high risk flagging
    - **Property 38: High risk flagging thresholds**
    - **Validates: Requirements 19.5**

- [x] 38. Implement False Acceptance Rate calculator
  - [x] 38.1 Create FalseAcceptanceCalculator class
    - Track abstention vs response for non-existent entity queries
    - Classify responses as correct refusal or false acceptance
    - _Requirements: 20.1, 20.3, 20.4_
  - [x] 38.2 Implement FAR computation
    - Compute FAR = failed_abstentions / total_nonexistent_queries
    - _Requirements: 20.2_
  - [x] 38.3 Write property test for FAR
    - **Property 39: False Acceptance Rate computation**
    - **Validates: Requirements 20.1, 20.2, 20.3, 20.4**

- [x] 39. Integrate hallucination metrics into EvaluationToolkit
  - [x] 39.1 Add hallucination profile to EvaluationResult
    - Include HallucinationProfile in evaluation results
    - _Requirements: 19.1_
  - [x] 39.2 Update ReportGenerator for hallucination metrics
    - Add hallucination metrics to JSON/CSV/text reports
    - _Requirements: 8.1, 19.4_

- [x] 40. Checkpoint - Ensure all hallucination quantification tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 41. Create hallucination quantification examples and documentation
  - Write example showing MiHR/MaHR computation
  - Write example showing cross-model consensus analysis
  - Write example showing uncertainty quantification
  - Update README with hallucination metrics documentation
  - _Requirements: 15-20_

- [ ] 42. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

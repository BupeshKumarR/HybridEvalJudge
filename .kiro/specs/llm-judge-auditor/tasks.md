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

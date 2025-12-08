"""
Unit tests for the PromptManager component.
"""

import pytest

from llm_judge_auditor.components.prompt_manager import PromptManager


class TestPromptManager:
    """Test suite for PromptManager."""
    
    def test_initialization(self):
        """Test that PromptManager initializes correctly."""
        pm = PromptManager()
        assert pm is not None
        assert len(pm.list_tasks()) > 0
    
    def test_list_tasks(self):
        """Test listing available tasks."""
        pm = PromptManager()
        tasks = pm.list_tasks()
        
        assert "factual_accuracy" in tasks
        assert "pairwise_ranking" in tasks
        assert "bias_detection" in tasks
    
    def test_get_prompt_factual_accuracy(self):
        """Test getting a factual accuracy prompt with variable substitution."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "factual_accuracy",
            source_text="Paris is the capital of France.",
            candidate_output="Paris is the capital of Germany.",
            retrieved_context="No additional context."
        )
        
        assert "Paris is the capital of France." in prompt
        assert "Paris is the capital of Germany." in prompt
        assert "No additional context." in prompt
        assert "factual accuracy" in prompt.lower()
        assert "chain-of-thought" in prompt.lower()
    
    def test_get_prompt_pairwise_ranking(self):
        """Test getting a pairwise ranking prompt."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "pairwise_ranking",
            source_text="The Earth orbits the Sun.",
            candidate_a="The Earth revolves around the Sun.",
            candidate_b="The Sun revolves around the Earth."
        )
        
        assert "The Earth orbits the Sun." in prompt
        assert "The Earth revolves around the Sun." in prompt
        assert "The Sun revolves around the Earth." in prompt
        assert "Candidate A" in prompt
        assert "Candidate B" in prompt
        assert "chain-of-thought" in prompt.lower()
    
    def test_get_prompt_bias_detection(self):
        """Test getting a bias detection prompt."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "bias_detection",
            candidate_output="Some text to analyze for bias."
        )
        
        assert "Some text to analyze for bias." in prompt
        assert "bias" in prompt.lower()
        assert "stereotypes" in prompt.lower()
        assert "chain-of-thought" in prompt.lower()
    
    def test_get_prompt_unsupported_task(self):
        """Test that unsupported tasks raise ValueError."""
        pm = PromptManager()
        
        with pytest.raises(ValueError, match="Unsupported task"):
            pm.get_prompt("nonexistent_task")
    
    def test_get_prompt_missing_variable(self):
        """Test that missing variables raise ValueError."""
        pm = PromptManager()
        
        with pytest.raises(ValueError, match="Missing required variable"):
            pm.get_prompt("factual_accuracy", source_text="Only source provided")
    
    def test_customize_prompt(self):
        """Test customizing a prompt template."""
        pm = PromptManager()
        
        custom_template = "Custom: {source_text} vs {candidate_output}"
        pm.customize_prompt("factual_accuracy", custom_template)
        
        prompt = pm.get_prompt(
            "factual_accuracy",
            source_text="Source",
            candidate_output="Candidate"
        )
        
        assert prompt == "Custom: Source vs Candidate"
    
    def test_customize_new_task(self):
        """Test adding a custom template for a new task."""
        pm = PromptManager()
        
        custom_template = "New task: {input_text}"
        pm.customize_prompt("custom_task", custom_template)
        
        prompt = pm.get_prompt("custom_task", input_text="Test input")
        assert prompt == "New task: Test input"
    
    def test_get_template(self):
        """Test getting raw template without substitution."""
        pm = PromptManager()
        
        template = pm.get_template("factual_accuracy")
        
        assert "{source_text}" in template
        assert "{candidate_output}" in template
        assert "{retrieved_context}" in template
        assert "chain-of-thought" in template.lower()
    
    def test_get_template_unsupported_task(self):
        """Test that getting template for unsupported task raises ValueError."""
        pm = PromptManager()
        
        with pytest.raises(ValueError, match="Unsupported task"):
            pm.get_template("nonexistent_task")
    
    def test_load_templates_not_implemented(self):
        """Test that load_templates raises NotImplementedError."""
        pm = PromptManager()
        
        with pytest.raises(NotImplementedError):
            pm.load_templates("config.yaml")
    
    def test_prompt_includes_instructions(self):
        """Test that prompts include clear instructions."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "factual_accuracy",
            source_text="Test",
            candidate_output="Test",
            retrieved_context=""
        )
        
        # Check for key instruction elements
        assert "Instructions:" in prompt or "Task:" in prompt
        assert "REASONING:" in prompt
        assert "SCORE:" in prompt or "WINNER:" in prompt
    
    def test_criterion_parameter_accepted(self):
        """Test that criterion parameter is accepted (for future extension)."""
        pm = PromptManager()
        
        # Should not raise an error even though criterion is not used yet
        prompt = pm.get_prompt(
            "factual_accuracy",
            criterion="completeness",
            source_text="Test",
            candidate_output="Test",
            retrieved_context=""
        )
        
        assert prompt is not None
    
    def test_empty_retrieved_context(self):
        """Test handling of empty retrieved context."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "factual_accuracy",
            source_text="Source text",
            candidate_output="Candidate output",
            retrieved_context=""
        )
        
        assert "Source text" in prompt
        assert "Candidate output" in prompt
    
    def test_pairwise_ranking_output_format(self):
        """Test that pairwise ranking prompt specifies output format."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "pairwise_ranking",
            source_text="Test",
            candidate_a="A",
            candidate_b="B"
        )
        
        assert "WINNER:" in prompt
        assert "A, B, or TIE" in prompt or "TIE" in prompt
    
    def test_bias_detection_severity_levels(self):
        """Test that bias detection prompt includes severity levels."""
        pm = PromptManager()
        
        prompt = pm.get_prompt(
            "bias_detection",
            candidate_output="Test text"
        )
        
        assert "LOW" in prompt
        assert "MEDIUM" in prompt
        assert "HIGH" in prompt
        assert "severity" in prompt.lower()

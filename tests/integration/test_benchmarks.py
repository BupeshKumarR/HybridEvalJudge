"""
Integration tests for benchmark validation.

These tests verify that the benchmark validation infrastructure works correctly.
"""

import json
import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from download_benchmarks import BenchmarkDownloader
from run_benchmarks import BenchmarkEvaluator, BenchmarkResult


class TestBenchmarkDownloader:
    """Tests for benchmark dataset downloading."""
    
    def test_downloader_initialization(self, tmp_path):
        """Test that downloader initializes correctly."""
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        assert downloader.output_dir == tmp_path
        assert tmp_path.exists()
    
    def test_download_fever(self, tmp_path):
        """Test FEVER dataset download."""
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        downloader.download_fever()
        
        # Check that files were created
        fever_dir = tmp_path / "fever"
        assert fever_dir.exists()
        assert (fever_dir / "train.jsonl").exists()
        assert (fever_dir / "dev.jsonl").exists()
        assert (fever_dir / "test.jsonl").exists()
        assert (fever_dir / "README.md").exists()
        
        # Check that files contain valid JSON
        with open(fever_dir / "dev.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                assert "id" in data
                assert "claim" in data
                assert "label" in data
                assert "evidence" in data
    
    def test_download_truthfulqa(self, tmp_path):
        """Test TruthfulQA dataset download."""
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        downloader.download_truthfulqa()
        
        # Check that files were created
        truthfulqa_dir = tmp_path / "truthfulqa"
        assert truthfulqa_dir.exists()
        assert (truthfulqa_dir / "truthfulqa.json").exists()
        assert (truthfulqa_dir / "README.md").exists()
        
        # Check that file contains valid JSON
        with open(truthfulqa_dir / "truthfulqa.json", "r") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            
            # Check structure of first item
            item = data[0]
            assert "id" in item
            assert "question" in item
            assert "best_answer" in item
            assert "correct_answers" in item
            assert "incorrect_answers" in item
    
    def test_download_all(self, tmp_path):
        """Test downloading all benchmarks."""
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        downloader.download_all()
        
        # Check that both datasets were created
        assert (tmp_path / "fever").exists()
        assert (tmp_path / "truthfulqa").exists()


class TestBenchmarkEvaluator:
    """Tests for benchmark evaluation."""
    
    @pytest.fixture
    def setup_benchmarks(self, tmp_path):
        """Set up benchmark datasets for testing."""
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        downloader.download_all()
        return tmp_path
    
    def test_evaluator_initialization(self, setup_benchmarks):
        """Test that evaluator initializes correctly."""
        evaluator = BenchmarkEvaluator(
            preset="balanced",
            benchmarks_dir=str(setup_benchmarks)
        )
        assert evaluator.preset == "balanced"
        assert evaluator.benchmarks_dir == setup_benchmarks
    
    def test_evaluate_fever(self, setup_benchmarks):
        """Test FEVER evaluation."""
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(setup_benchmarks)
        )
        
        # Run evaluation with limited samples
        result = evaluator.evaluate_fever(max_samples=5)
        
        # Check result structure
        assert isinstance(result, BenchmarkResult)
        assert result.dataset == "FEVER"
        assert result.total_samples == 5
        assert result.correct + result.incorrect + result.errors == result.total_samples
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1_score <= 1
        assert result.avg_confidence >= 0
        assert result.avg_latency_ms >= 0
    
    def test_evaluate_truthfulqa(self, setup_benchmarks):
        """Test TruthfulQA evaluation."""
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(setup_benchmarks)
        )
        
        # Run evaluation with limited samples
        result = evaluator.evaluate_truthfulqa(max_samples=3)
        
        # Check result structure
        assert isinstance(result, BenchmarkResult)
        assert result.dataset == "TruthfulQA"
        assert result.total_samples == 3
        assert result.correct + result.incorrect + result.errors == result.total_samples
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1_score <= 1
    
    def test_save_results(self, setup_benchmarks, tmp_path):
        """Test saving benchmark results."""
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(setup_benchmarks)
        )
        
        # Run evaluations
        fever_result = evaluator.evaluate_fever(max_samples=5)
        truthfulqa_result = evaluator.evaluate_truthfulqa(max_samples=3)
        
        # Save results
        output_file = tmp_path / "test_results.json"
        evaluator.save_results([fever_result, truthfulqa_result], str(output_file))
        
        # Check that file was created
        assert output_file.exists()
        
        # Load and verify results
        with open(output_file, "r") as f:
            loaded_results = json.load(f)
        
        assert len(loaded_results) == 2
        assert loaded_results[0]["dataset"] == "FEVER"
        assert loaded_results[1]["dataset"] == "TruthfulQA"
    
    def test_compare_to_baseline(self, setup_benchmarks, capsys):
        """Test baseline comparison."""
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(setup_benchmarks)
        )
        
        # Run evaluations
        fever_result = evaluator.evaluate_fever(max_samples=5)
        truthfulqa_result = evaluator.evaluate_truthfulqa(max_samples=3)
        
        # Compare to baseline
        evaluator.compare_to_baseline([fever_result, truthfulqa_result])
        
        # Check that output was printed
        captured = capsys.readouterr()
        assert "Comparison to Baselines" in captured.out
        assert "FEVER" in captured.out
        assert "TruthfulQA" in captured.out
    
    def test_missing_dataset_error(self, tmp_path):
        """Test that missing dataset raises appropriate error."""
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(tmp_path)
        )
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_fever()
    
    def test_max_samples_limit(self, setup_benchmarks):
        """Test that max_samples parameter works correctly."""
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(setup_benchmarks)
        )
        
        # Evaluate with different sample limits
        result_2 = evaluator.evaluate_fever(max_samples=2)
        result_5 = evaluator.evaluate_fever(max_samples=5)
        
        assert result_2.total_samples == 2
        assert result_5.total_samples == 5


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a BenchmarkResult."""
        result = BenchmarkResult(
            dataset="TEST",
            total_samples=100,
            correct=85,
            incorrect=15,
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            avg_confidence=0.89,
            avg_latency_ms=245.3,
            errors=0
        )
        
        assert result.dataset == "TEST"
        assert result.total_samples == 100
        assert result.correct == 85
        assert result.accuracy == 0.85
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        from dataclasses import asdict
        
        result = BenchmarkResult(
            dataset="TEST",
            total_samples=100,
            correct=85,
            incorrect=15,
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            avg_confidence=0.89,
            avg_latency_ms=245.3,
            errors=0
        )
        
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["dataset"] == "TEST"
        assert result_dict["accuracy"] == 0.85


class TestBenchmarkIntegration:
    """Integration tests for full benchmark workflow."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow: download -> evaluate -> save."""
        # Download benchmarks
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        downloader.download_all()
        
        # Run evaluations
        evaluator = BenchmarkEvaluator(
            preset="fast",
            benchmarks_dir=str(tmp_path)
        )
        
        fever_result = evaluator.evaluate_fever(max_samples=5)
        truthfulqa_result = evaluator.evaluate_truthfulqa(max_samples=3)
        
        # Save results
        output_file = tmp_path / "results.json"
        evaluator.save_results([fever_result, truthfulqa_result], str(output_file))
        
        # Verify everything worked
        assert output_file.exists()
        
        with open(output_file, "r") as f:
            results = json.load(f)
        
        assert len(results) == 2
        assert all(r["total_samples"] > 0 for r in results)
        assert all(0 <= r["accuracy"] <= 1 for r in results)
    
    def test_multiple_presets(self, tmp_path):
        """Test running benchmarks with different presets."""
        # Download benchmarks
        downloader = BenchmarkDownloader(output_dir=str(tmp_path))
        downloader.download_fever()
        
        presets = ["fast", "balanced"]
        results = {}
        
        for preset in presets:
            evaluator = BenchmarkEvaluator(
                preset=preset,
                benchmarks_dir=str(tmp_path)
            )
            result = evaluator.evaluate_fever(max_samples=5)
            results[preset] = result
        
        # Both presets should produce valid results
        assert all(r.total_samples == 5 for r in results.values())
        assert all(0 <= r.accuracy <= 1 for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

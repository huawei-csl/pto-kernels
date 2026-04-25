from pathlib import Path

from pto_kernels.bench import BenchmarkRunner, load_spec


def test_seed_spec_loads_with_defaults():
    spec = load_spec(Path("bench/specs/attention/flash_attention_score.yaml"))
    assert spec.device["soc"] == "ascend910b"
    assert spec.bench.warmup == 20
    assert spec.correctness.shape_sets == ["smoke", "nominal", "boundary"]


def test_benchmark_runner_dry_run_emits_report(tmp_path):
    runner = BenchmarkRunner(results_dir=tmp_path)
    report = runner.run("bench/specs/gmm/grouped_matmul.yaml", dry_run=True)
    assert report["dry_run"] is True
    assert report["family"] == "gmm"
    assert report["pto"]["adapter"] == "bench/adapters/ptodsl/gmm/grouped_matmul.py"

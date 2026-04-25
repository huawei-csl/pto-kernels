from pathlib import Path

from pto_kernels.utils.env import parse_npu_smi_output


def test_parse_npu_smi_output_for_910b_sample():
    sample = (Path(__file__).parent / "data" / "npu_smi_910b.txt").read_text(encoding="utf-8")
    model, count = parse_npu_smi_output(sample)
    assert model == "910B1"
    assert count == 2

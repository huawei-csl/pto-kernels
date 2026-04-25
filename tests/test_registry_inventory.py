from pto_kernels.registry import included_kernel_records, kernel_counts


def test_kernel_inventory_counts_match_plan():
    counts = kernel_counts()
    assert counts["included"] == 110
    assert counts["excluded_ai_cpu"] == 2
    assert counts["excluded_a3_only"] == 11
    assert counts["seed_kernels"] == 6


def test_included_kernel_inventory_contains_seed_and_wave_data():
    records = included_kernel_records()
    names = {record.name for record in records}
    assert "apply_rotary_pos_emb" in names
    assert "matmul_reduce_scatter" in names
    assert "engram_gate_bwd" in names
    assert any(record.wave == "wave5" for record in records if record.name == "matmul_reduce_scatter")


def test_readme_kernel_inventory_matches_registered_status():
    from pathlib import Path

    import yaml

    readme = Path("README.md").read_text(encoding="utf-8")
    data = yaml.safe_load(Path("bench/kernel_inventory.yaml").read_text(encoding="utf-8"))
    missing = []
    for item in data["included"]:
        row = f"| `{item['family']}/{item['name']}` |"
        status = f"| `{item['status']}` |"
        if row not in readme or status not in readme:
            missing.append(f"{item['family']}/{item['name']}")
    for reason, items in data["excluded"].items():
        status = f"| `excluded_{reason}` |"
        for item in items:
            row = f"| `{item['family']}/{item['name']}` |"
            if row not in readme or status not in readme:
                missing.append(f"{item['family']}/{item['name']}")
    assert not missing

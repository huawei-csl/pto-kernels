from pathlib import Path

import yaml


def test_cann_recipes_reference_is_pinned():
    manifest = yaml.safe_load(Path("external/manifest.lock").read_text(encoding="utf-8"))
    recipe = manifest["repos"]["cann-recipes-infer"]
    assert recipe["url"] == "https://gitcode.com/cann/cann-recipes-infer.git"
    assert recipe["commit"] == "377f20f62d86b3da882b5084b46e02c735e619a3"


def test_kernel_writer_surface_is_present():
    required_paths = [
        Path("docs/cann_recipes_infer_notes.md"),
        Path("skills/pto-kernel-writer/SKILL.md"),
        Path("templates/kernel_writer/README.md"),
        Path("templates/kernel_writer/kernel.py.template"),
        Path("templates/kernel_writer/meta.py.template"),
        Path("templates/kernel_writer/bench_spec.yaml.template"),
        Path("templates/kernel_writer/adapter_baseline.py.template"),
        Path("templates/kernel_writer/adapter_ptodsl.py.template"),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    assert not missing

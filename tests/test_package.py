from pathlib import Path

from scripts.package_submission import submission_files


def test_package_allowlist_excludes_repository_material():
    root = Path(__file__).resolve().parents[1]
    files = submission_files(root)
    names = {path.relative_to(root).as_posix() for path in files}

    assert {"metadata.json", "run.py", "structured_features.py", "targeted_route.py"} <= names
    assert any(name.startswith("models/reranker/") for name in names)
    assert any(name.startswith("models/adapted_reranker/") for name in names)
    assert any(name.startswith("models/ambiguous_reranker/") for name in names)
    assert any(name.startswith("models/weak_reranker/") for name in names)
    assert not any(name.startswith(("docs/", "tests/", "training/")) for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

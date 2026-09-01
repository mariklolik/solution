import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ("run.py", "structured_features.py", "targeted_route.py")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_official_entrypoint_contract():
    metadata = json.loads((ROOT / "metadata.json").read_text())

    assert metadata == {
        "image": "odsai/ecup26-matching-baseline:1.0",
        "entry_point": "python -u run.py",
    }


def test_runtime_has_no_network_clients():
    forbidden = {"httpx", "requests", "socket", "urllib", "wget"}
    imported = set()
    for name in RUNTIME_FILES:
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

    assert not imported & forbidden


def test_scored_runtime_hashes():
    manifest = json.loads((ROOT / "artifacts" / "scored-artifact.json").read_text())

    for relative, expected in manifest["files"].items():
        assert sha256(ROOT / relative) == expected


def test_model_files_stay_outside_git():
    ignored = (ROOT / ".gitignore").read_text().splitlines()

    assert "models/" in ignored

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.package_submission import submission_files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_repository(root):
    root = Path(root)
    receipt = json.loads((root / "artifacts" / "scored-artifact.json").read_text())
    failures = {}
    for relative, expected in receipt["files"].items():
        actual = sha256(root / relative)
        if actual != expected:
            failures[relative] = {"expected": expected, "actual": actual}
    expected_files = set(receipt["files"])
    packaged_files = {path.relative_to(root).as_posix() for path in submission_files(root)}
    if packaged_files != expected_files:
        failures["package_allowlist"] = {
            "missing": sorted(expected_files - packaged_files),
            "unexpected": sorted(packaged_files - expected_files),
        }
    if failures:
        raise RuntimeError(json.dumps(failures, ensure_ascii=False, indent=2))
    return {"files": len(packaged_files), "status": "verified"}


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        failed = archive.testzip()
        if failed:
            raise RuntimeError(f"CRC failure: {failed}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate archive members")
    return {"members": len(names), "status": "verified"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--zip")
    args = parser.parse_args()
    result = {"repository": verify_repository(args.root)}
    if args.zip:
        result["zip"] = verify_zip(args.zip)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

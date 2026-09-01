import argparse
import zipfile
from pathlib import Path


RUNTIME_FILES = ("metadata.json", "run.py", "structured_features.py", "targeted_route.py")
MODEL_DIRECTORIES = ("reranker", "adapted_reranker", "ambiguous_reranker", "weak_reranker")


def submission_files(root):
    root = Path(root)
    files = [root / name for name in RUNTIME_FILES]
    for directory in MODEL_DIRECTORIES:
        files.extend(
            path
            for path in sorted((root / "models" / directory).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(map(str, missing)))
    return files


def package(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in submission_files(root):
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        failed = archive.testzip()
        if failed:
            raise RuntimeError(f"CRC failure: {failed}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(package(args.root, args.output))


if __name__ == "__main__":
    main()

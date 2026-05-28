from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run(command: list[object]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"$ {printable}")
    subprocess.run([str(part) for part in command], cwd=REPO_ROOT, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="chemical_embedding_smoke_") as tmp:
        work = Path(tmp)
        run([PYTHON, "-m", "chemical_embedding_pipeline.check_coverage"])
        run(
            [
                PYTHON,
                "-m",
                "chemical_embedding_pipeline.generate_lfc_embeddings",
                "--dataset",
                "tahoe",
                "--steps",
                "metadata",
                "random",
                "--output-dir",
                work,
                "--verify",
                "--merge",
            ]
        )
        summary = work / "tahoe_verification_summary.csv"
        merged = work / "tahoe_lfc_embeddings.h5ad"
        if not summary.exists():
            raise AssertionError(f"Expected summary was not written: {summary}")
        if not merged.exists():
            raise AssertionError(f"Expected merged H5AD was not written: {merged}")
        text = summary.read_text()
        for expected in ["moa_onehot", "target_onehot", "random"]:
            if expected not in text:
                raise AssertionError(f"{expected} missing from smoke summary.")
        print(f"Smoke test passed: {summary}")


if __name__ == "__main__":
    main()

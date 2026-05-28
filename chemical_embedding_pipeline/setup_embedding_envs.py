from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PY311 = "3.11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create optional side environments for the simplified LFC embedding generator."
    )
    parser.add_argument(
        "--env",
        nargs="+",
        choices=["legacy-secfp", "minimol", "pretrained-molfeat", "all"],
        default=["minimol"],
        help=(
            "Which side environment(s) to create. Defaults to MiniMol only; "
            "legacy-secfp and pretrained-molfeat are optional parity/dependency paths."
        ),
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_PY311,
        help="Python 3.11 executable to use for side environments.",
    )
    parser.add_argument(
        "--uv-cache-dir",
        type=Path,
        default=Path(".uv-cache"),
        help="uv cache directory, relative to the repository root unless absolute.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], uv_cache_dir: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    if dry_run:
        return
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(uv_cache_dir)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def ensure_venv(path: Path, python: str, uv_cache_dir: Path, dry_run: bool) -> None:
    if (REPO_ROOT / path / "bin" / "python").exists():
        print(f"Using existing virtual environment at: {path}")
        return
    run(
        [
            "uv",
            "venv",
            str(path),
            "--python",
            str(python),
        ],
        uv_cache_dir,
        dry_run,
    )


def selected_envs(values: list[str]) -> set[str]:
    if "all" in values:
        return {"legacy-secfp", "minimol", "pretrained-molfeat"}
    return set(values)


def setup_legacy_secfp(python: str, uv_cache_dir: Path, dry_run: bool) -> None:
    ensure_venv(Path(".chemembed-legacy-py311"), python, uv_cache_dir, dry_run)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            ".chemembed-legacy-py311/bin/python",
            "anndata<0.11",
            "numpy<2",
            "pandas<2",
            "scipy",
            "scikit-learn",
            "rdkit==2023.9.6",
            "molfeat==0.10.1",
            "mordredcommunity",
        ],
        uv_cache_dir,
        dry_run,
    )


def setup_minimol(python: str, uv_cache_dir: Path, dry_run: bool) -> None:
    ensure_venv(Path(".chemembed-minimol-py311"), python, uv_cache_dir, dry_run)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            ".chemembed-minimol-py311/bin/python",
            "torch",
        ],
        uv_cache_dir,
        dry_run,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            ".chemembed-minimol-py311/bin/python",
            "--no-build-isolation",
            "anndata<0.11",
            "numpy<2",
            "pandas<2.3",
            "scipy==1.11.4",
            "scikit-learn",
            "minimol",
        ],
        uv_cache_dir,
        dry_run,
    )


def setup_pretrained_molfeat(python: str, uv_cache_dir: Path, dry_run: bool) -> None:
    ensure_venv(Path(".chemembed-pretrained-molfeat-py311"), python, uv_cache_dir, dry_run)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            ".chemembed-pretrained-molfeat-py311/bin/python",
            "anndata<0.11",
            "numpy<2",
            "pandas<2.3",
            "scipy",
            "scikit-learn",
            "rdkit",
            "molfeat[dgl]",
            "dgllife",
            "dgl==2.1.0",
            "torch==2.2.1",
            "torchdata==0.7.1",
        ],
        uv_cache_dir,
        dry_run,
    )


def main() -> None:
    args = parse_args()
    uv_cache_dir = args.uv_cache_dir
    if not uv_cache_dir.is_absolute():
        uv_cache_dir = REPO_ROOT / uv_cache_dir

    envs = selected_envs(args.env)
    if "legacy-secfp" in envs:
        setup_legacy_secfp(args.python, uv_cache_dir, args.dry_run)
    if "minimol" in envs:
        setup_minimol(args.python, uv_cache_dir, args.dry_run)
    if "pretrained-molfeat" in envs:
        setup_pretrained_molfeat(args.python, uv_cache_dir, args.dry_run)


if __name__ == "__main__":
    main()

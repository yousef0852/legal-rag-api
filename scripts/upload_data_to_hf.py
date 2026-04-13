#!/usr/bin/env python3
"""Create a private Hugging Face Dataset repo and upload local data (full `data/` or `data/markdown` only)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami
from huggingface_hub.errors import LocalTokenNotFoundError


def _load_dotenv(path: Path) -> None:
    """Set missing os.environ keys from a simple KEY=VALUE .env file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    _load_dotenv(root / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Upload project data/markdown into path markdown/ in the repo (overrides --data-dir default).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Local folder to upload (default: project data/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("HF_REPO_ID", ""),
        help="Full id, e.g. username/my-dataset. If omitted, uses HF_REPO_ID or <your_user>/HF_REPO_NAME.",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=os.environ.get("HF_REPO_NAME", "legal-rag-api-data"),
        help="Dataset name when --repo-id is not set (default: legal-rag-api-data).",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Destination path inside the repo (default: repo root).",
    )
    parser.add_argument(
        "--large-folder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HfApi.upload_large_folder (resumable, batched commits). Default: on. Use --no-large-folder for a single upload_folder.",
    )
    parser.add_argument(
        "--batch-subdirs",
        action="store_true",
        help="With --no-large-folder: upload each top-level folder under the source dir as a separate upload_folder (smaller commits).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Workers for upload_large_folder (default: HF picks from CPU count). Lower helps slow networks.",
    )
    args = parser.parse_args()

    if args.markdown_only:
        data_dir = (root / "data" / "markdown").resolve()
        path_in_repo = args.path_in_repo.strip() if args.path_in_repo.strip() else "markdown"
    else:
        data_dir = args.data_dir.resolve()
        path_in_repo = args.path_in_repo.strip() or None

    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.markdown_only and args.large_folder and path_in_repo != "markdown":
        print(
            "upload_large_folder keeps paths as data/markdown/... -> markdown/... in the repo.\n"
            "Omit --path-in-repo or use markdown. For a custom prefix, run with --no-large-folder --batch-subdirs.",
            file=sys.stderr,
        )
        return 1

    try:
        me = whoami()
    except LocalTokenNotFoundError:
        print(
            "No Hugging Face token found. Add HF_TOKEN to .env in the project root, or:\n"
            "  export HF_TOKEN=hf_...\n"
            "or run: hf auth login\n"
            "Create tokens at https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    repo_id = args.repo_id.strip()
    if not repo_id:
        name = me.get("name")
        if not name:
            print("Could not resolve Hub username; set --repo-id or log in.", file=sys.stderr)
            return 1
        repo_id = f"{name}/{args.repo_name.strip()}"

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    ignore_extra = ["**/.DS_Store", ".DS_Store"]

    if args.large_folder:
        if args.markdown_only:
            # Paths in repo: markdown/... (relative to data/)
            folder_path = root / "data"
            allow_patterns = "markdown/**"
        else:
            folder_path = data_dir
            allow_patterns = None
        kwargs = {
            "repo_id": repo_id,
            "folder_path": str(folder_path.resolve()),
            "repo_type": "dataset",
            "private": True,
            "ignore_patterns": ignore_extra,
        }
        if allow_patterns is not None:
            kwargs["allow_patterns"] = allow_patterns
        if args.num_workers is not None:
            kwargs["num_workers"] = args.num_workers
        api.upload_large_folder(**kwargs)
    elif args.batch_subdirs:
        base = (path_in_repo or "").strip().strip("/")
        subs = sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        for sub in subs:
            dest = f"{base}/{sub.name}" if base else sub.name
            print(f"Batch: {sub.name} -> {dest}/", flush=True)
            api.upload_folder(
                folder_path=str(sub),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=dest,
                ignore_patterns=ignore_extra,
            )
        root_files = [p for p in data_dir.iterdir() if p.is_file() and p.name != ".DS_Store"]
        for fp in root_files:
            dest_path = f"{base}/{fp.name}" if base else fp.name
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=dest_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
    else:
        api.upload_folder(
            folder_path=str(data_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            ignore_patterns=ignore_extra,
        )

    print(f"Uploaded {data_dir} to https://huggingface.co/datasets/{repo_id} (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

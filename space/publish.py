"""Create and publish the Hugging Face Space.

Usage:

    pip install huggingface_hub
    python space/publish.py --user YOUR_HF_USERNAME

The script is idempotent: run it again after any change to redeploy. It reads
the Gemini key from your local .env and installs it as a Space secret, so the
key is never written into the Space repository.

Authentication comes from `huggingface-cli login` or the HF_TOKEN environment
variable. The token needs write access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPACE_DIR = Path(__file__).parent
PROJECT_ROOT = SPACE_DIR.parent


def read_local_key() -> str:
    """Pull GEMINI_API_KEY out of the project's .env, if present."""
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return ""
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            return "" if value in ("", "your-key-here", "PASTE_YOUR_KEY_HERE") else value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the RepoSage Space.")
    parser.add_argument("--user", required=True, help="Your Hugging Face username.")
    parser.add_argument("--name", default="reposage", help="Space name (default: reposage).")
    parser.add_argument("--private", action="store_true", help="Create it private.")
    parser.add_argument("--no-secret", action="store_true", help="Skip uploading the API key.")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed. Run:  pip install huggingface_hub")
        return 1

    repo_id = f"{args.user}/{args.name}"
    api = HfApi()

    try:
        whoami = api.whoami()
    except Exception:
        print("Not authenticated. Run `huggingface-cli login`, or set HF_TOKEN.")
        return 1
    print(f"Authenticated as {whoami.get('name', 'unknown')}")

    print(f"Creating or reusing Space {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    if not args.no_secret:
        key = read_local_key()
        if key:
            api.add_space_secret(repo_id=repo_id, key="GEMINI_API_KEY", value=key)
            print("Installed GEMINI_API_KEY as a Space secret.")
        else:
            print(
                "No usable GEMINI_API_KEY found in .env. Add it manually at\n"
                f"  https://huggingface.co/spaces/{repo_id}/settings\n"
                "under 'Variables and secrets', or the demo cannot answer questions."
            )

    print("Uploading ...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        # publish.py is tooling, not part of the running Space.
        ignore_patterns=["publish.py", "__pycache__/*", "*.pyc"],
        commit_message="Deploy RepoSage demo",
    )

    _link_readme(repo_id)

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDone. The Space is building at:\n  {url}")
    print("\nThe first build takes a few minutes. Watch the Logs tab there.")
    print(f"Once it is running, the live demo is at:\n  https://{args.user}-{args.name}.hf.space")
    return 0


def _link_readme(repo_id: str) -> None:
    """Point the GitHub README at the Space that now exists.

    The README ships with a placeholder because the Space id is not known until
    somebody publishes one. Rewriting it here means the repository can never
    advertise a demo URL that does not resolve.
    """
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if "HF_USERNAME" not in text:
        return
    readme.write_text(text.replace("HF_USERNAME/reposage", repo_id), encoding="utf-8")
    print(f"Updated README.md demo links to {repo_id}. Commit and push the change.")


if __name__ == "__main__":
    sys.exit(main())

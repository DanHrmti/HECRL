from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "DanHrmti/hecrl_visual_encoders"


def main() -> None:
    """Download all repo files except README.md and .gitattributes into visual_encoders/chkpts."""
    target_dir = Path(__file__).resolve().parent / "chkpts"

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(target_dir),
        ignore_patterns=["README.md", ".gitattributes"],
    )


if __name__ == "__main__":
    main()

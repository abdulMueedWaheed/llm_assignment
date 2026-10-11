import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from output_utils import print_field, print_header, print_section


def load_gdown():
    try:
        import gdown
    except ImportError:
        print("Error: 'gdown' package is missing.")
        print("Install it with: python -m pip install gdown")
        raise SystemExit(1)
    return gdown


# Shared Google Drive URL containing the project datasets
DEFAULT_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1OhQf1P5FBL99GPeuY9nu363Xjbu415TQ"
)


def get_default_dest_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "data" / "processed"


def scan_remote_drive_folder(
    folder_url: str,
    target_subfolder: str = "processed",
) -> list[dict]:
    gdown = load_gdown()
    """
    Queries Google Drive directly to list all files in the shared folder.
    Filters files in the target subfolder (for example, 'processed').
    No file names or IDs are hardcoded.
    """
    print_section("GOOGLE DRIVE SEARCH")
    print_field("Drive folder", folder_url)
    print_field("Target subfolder", target_subfolder)
    print("Connecting to Google Drive...")

    # Use skip_download=True to list files and metadata without downloading
    try:
        drive_items = gdown.download_folder(
            url=folder_url,
            quiet=True,
            skip_download=True,
        )
    except Exception as exc:
        print(f"Failed to scan Google Drive folder: {exc}")
        return []

    discovered_files = []
    print_section("MATCHING FILES")

    for item in drive_items:
        relative_path = str(getattr(item, "path", ""))
        file_id = getattr(item, "id", "")
        file_name = Path(relative_path).name

        # Check if this file lives in the target subfolder (e.g., 'processed')
        parts = Path(relative_path).parts
        if target_subfolder in parts:
            print(f"Found: {file_name} ({relative_path})")
            discovered_files.append({
                "name": file_name,
                "id": file_id,
                "remote_path": relative_path,
            })

    return discovered_files


def download_remote_file(file_info: dict, dest_dir: Path, force: bool = False) -> bool:
    """
    Downloads a single file from Google Drive into the local destination folder.
    Skips download if file already exists with non-zero size (unless force=True).
    """
    file_name = file_info["name"]
    gdown = load_gdown()
    file_id = file_info["id"]
    target_path = dest_dir / file_name

    if target_path.exists() and target_path.stat().st_size > 0 and not force:
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"Already present: {file_name} ({size_mb:.2f} MB); skipping download.")
        return True

    print_section(f"DOWNLOADING {file_name}")
    print_field("Destination", target_path)

    download_url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(
            url=download_url,
            output=str(target_path),
            quiet=False,
            resume=True,
        )
        if target_path.exists() and target_path.stat().st_size > 0:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"Download complete: {file_name} ({size_mb:.2f} MB).")
            return True
        else:
            print(f"Download did not complete: {file_name}.")
            return False
    except Exception as exc:
        print(f"Error downloading {file_name}: {exc}")
        return False


def verify_local_files(dest_dir: Path) -> None:
    """
    Lists all text files currently saved in the local destination directory.
    """
    print_section("LOCAL DATA FILES")
    print_field("Destination folder", dest_dir.resolve())

    txt_files = list(dest_dir.glob("*.txt"))
    if not txt_files:
        print("  No .txt files found in this directory.")
    else:
        for f in txt_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"{f.name:<24} {size_mb:>10.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan Google Drive and download files from the processed subfolder."
        )
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DRIVE_FOLDER_URL,
        help="Shared Google Drive folder link",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="processed",
        help="Drive subfolder to filter files by (default: processed)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(get_default_dest_dir()),
        help="Local destination folder "
        "(default: llm_assignment/data/processed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist locally",
    )

    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print_header("PROCESSED CORPUS DOWNLOAD")
    print_field("Local destination", dest_dir.resolve())

    # 1. Dynamically read file names and IDs from Google Drive
    remote_files = scan_remote_drive_folder(args.url, target_subfolder=args.subfolder)

    if not remote_files:
        print(f"No files found in the '{args.subfolder}' Drive subfolder.")
        return

    print_section("DOWNLOAD PLAN")
    print_field("Files found", len(remote_files))

    # 2. Download each discovered file
    for item in remote_files:
        download_remote_file(item, dest_dir, force=args.force)

    # 3. Print final status of files in destination folder
    verify_local_files(dest_dir)


if __name__ == "__main__":
    main()

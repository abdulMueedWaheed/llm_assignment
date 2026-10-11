import argparse
from pathlib import Path

from output_utils import show_section, show_title, show_value

try:
    import gdown
except ImportError:
    print("The gdown package is missing. Install it with: pip install gdown")
    raise SystemExit(1)


DEFAULT_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1OhQf1P5FBL99GPeuY9nu363Xjbu415TQ"
)


def find_processed_files(
    folder_url: str,
    target_folder: str,
) -> list[dict[str, str]]:
    show_section("SEARCHING GOOGLE DRIVE")
    show_value("Drive folder", folder_url)
    show_value("Target folder", target_folder)

    try:
        drive_files = gdown.download_folder(
            url=folder_url,
            quiet=True,
            skip_download=True,
        )
    except Exception as error:
        print(f"Could not search Google Drive: {error}")
        return []

    matching_files = []
    for drive_file in drive_files:
        remote_path = str(getattr(drive_file, "path", ""))
        if target_folder not in Path(remote_path).parts:
            continue

        matching_files.append(
            {
                "name": Path(remote_path).name,
                "id": str(getattr(drive_file, "id", "")),
                "remote_path": remote_path,
            }
        )

    return matching_files


def download_one_file(
    file_info: dict[str, str],
    destination: Path,
    force_download: bool,
) -> bool:
    local_path = destination / file_info["name"]
    if (
        local_path.exists()
        and local_path.stat().st_size > 0
        and not force_download
    ):
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(
            f"Already downloaded: {local_path.name} "
            f"({size_mb:.2f} MB)"
        )
        return True

    show_section(f"DOWNLOADING {local_path.name}")
    show_value("Save location", local_path)
    download_url = f"https://drive.google.com/uc?id={file_info['id']}"

    try:
        gdown.download(
            url=download_url,
            output=str(local_path),
            quiet=False,
            resume=True,
        )
    except Exception as error:
        print(f"Download failed for {local_path.name}: {error}")
        return False

    if not local_path.exists() or local_path.stat().st_size == 0:
        print(f"Download did not create a file: {local_path.name}")
        return False

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"Download complete: {local_path.name} ({size_mb:.2f} MB)")
    return True


def show_local_text_files(destination: Path) -> None:
    show_section("LOCAL TEXT FILES")
    show_value("Folder", destination.resolve())
    text_files = sorted(destination.glob("*.txt"))

    if not text_files:
        print("No text files found.")
        return

    for text_file in text_files:
        size_mb = text_file.stat().st_size / (1024 * 1024)
        print(f"{text_file.name:<24} {size_mb:>8.2f} MB")


def main() -> int:
    project_folder = Path(__file__).resolve().parent.parent
    default_destination = project_folder / "data" / "processed"
    parser = argparse.ArgumentParser(
        description="Download prepared corpus text files from Google Drive."
    )
    parser.add_argument("--url", default=DEFAULT_DRIVE_URL)
    parser.add_argument("--subfolder", default="processed")
    parser.add_argument("--dest", default=str(default_destination))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)

    show_title("PROCESSED CORPUS DOWNLOAD")
    show_value("Save folder", destination.resolve())
    files = find_processed_files(args.url, args.subfolder)

    if not files:
        print(f"No files were found in the '{args.subfolder}' folder.")
        return 1

    show_section("FILES TO DOWNLOAD")
    show_value("Number of files", len(files))

    all_downloads_succeeded = True
    for file_info in files:
        succeeded = download_one_file(
            file_info,
            destination,
            args.force,
        )
        all_downloads_succeeded = all_downloads_succeeded and succeeded

    show_local_text_files(destination)
    return 0 if all_downloads_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())

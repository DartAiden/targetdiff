from pathlib import Path
import tarfile


model_root = Path(__file__).resolve().parents[1]
archive_path = model_root / "targetdiff_source.tar.gz"
source_directories = ("datasets", "models", "scripts", "utils")

with tarfile.open(archive_path, "w:gz") as archive:
    for directory in source_directories:
        archive.add(model_root / directory, arcname=directory)

print("targetdiff/targetdiff_source.tar.gz")

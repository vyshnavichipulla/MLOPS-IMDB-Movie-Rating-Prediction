from pathlib import Path
import subprocess
import sys

dataset = Path("data/raw/IMDB-Movie-Data-selected-columns.csv")
backup = dataset.with_name(dataset.name + ".lab9_backup")

if not dataset.exists():
    raise FileNotFoundError(f"Dataset not found: {dataset}")

if backup.exists():
    raise FileExistsError(f"Backup already exists: {backup}")

try:
    dataset.rename(backup)
    result = subprocess.run(
        [sys.executable, "pipelines/run_pipeline.py"],
        check=False
    )
    if result.returncode == 0:
        raise RuntimeError("Expected failure was not detected.")
    print("Expected failure detected.")
finally:
    if backup.exists():
        backup.rename(dataset)

print("Dataset restored. Running recovery pipeline...")
subprocess.run(
    [sys.executable, "pipelines/run_pipeline.py"],
    check=True
)
print("Lab 9 recovery successful.")

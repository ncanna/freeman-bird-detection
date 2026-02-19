import urllib.request
import zipfile
from pathlib import Path

def setup_dataset(data_dir: Path, zip_url: str, dataset_name: str) -> None:
    """
    Download and extract a dataset to the specified directory.
    
    Args:
        data_dir:     Root data directory (e.g. Path to .../data/)
        zip_url:      URL to the dataset zip file
        dataset_name: Name of the dataset folder (e.g. 'african-wildlife')
    """
    images_dir = data_dir / dataset_name / "images"
    if images_dir.exists():
        print(f"Dataset already exists at {images_dir}")
        return
    
    data_dir = data_dir / dataset_name
    zip_path = data_dir / f"{dataset_name}.zip"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} to {zip_path}...")
    urllib.request.urlretrieve(zip_url, zip_path)

    print(f"Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    zip_path.unlink()
    print(f"Dataset ready at {images_dir}")
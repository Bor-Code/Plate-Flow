import argparse
import urllib.request
from pathlib import Path

from plateflow.config.logging import get_logger, setup_logging

logger = get_logger(__name__)


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} to {dest_path}")
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded {dest_path.name}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["yolo11n", "yolo11s"], 
        default="yolo11n",
        help="Type of YOLO model to download"
    )
    args = parser.parse_args()

    weights_dir = Path("weights")
    dest_path = weights_dir / f"{args.model_type}.pt"
    
    urls = {
        "yolo11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "yolo11s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    }
    
    download_file(urls[args.model_type], dest_path)


if __name__ == "__main__":
    main()

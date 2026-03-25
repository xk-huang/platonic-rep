#!/usr/bin/env python3
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_local_dataset(dataset_path: str, split: str):
    try:
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            first_split = next(iter(ds.keys()))
            print(f"Requested split '{split}' not found; using '{first_split}' instead.")
            return ds[first_split]
        return ds
    except Exception:
        # Fallback for dataset repos/scripts that are loadable via load_dataset.
        return load_dataset(dataset_path, split=split)


def pick_image_column(ds: Dataset, preferred_column: str) -> str:
    if preferred_column in ds.column_names:
        return preferred_column

    for col in ds.column_names:
        feature = ds.features.get(col)
        if feature is not None and feature.__class__.__name__ == "Image":
            return col
    if "image" in ds.column_names:
        return "image"
    raise ValueError(f"No image column found. Columns: {ds.column_names}")


def _save_single_image(image_obj, out_path: Path):
    # HF Image feature decodes to PIL.Image.Image by default.
    if hasattr(image_obj, "save"):
        image_obj.save(out_path)
        return

    # If decoding is disabled, image may be a dict with bytes/path.
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            out_path.write_bytes(image_obj["bytes"])
            return
        if image_obj.get("path"):
            src = Path(image_obj["path"])
            out_path.write_bytes(src.read_bytes())
            return

    raise TypeError(f"Unsupported image object type: {type(image_obj)}")


def save_example_image(image_obj, out_path: Path) -> int:
    if isinstance(image_obj, (list, tuple)):
        if len(image_obj) == 0:
            return 0
        for idx, item in enumerate(image_obj):
            list_out_path = out_path.with_name(f"{out_path.stem}_{idx:02d}{out_path.suffix}")
            _save_single_image(item, list_out_path)
        return len(image_obj)

    _save_single_image(image_obj, out_path)
    return 1


def main():
    parser = argparse.ArgumentParser(description="Export images from a local Hugging Face dataset.")
    parser.add_argument(
        "--dataset-path",
        default="/opt/dlami/nvme/xhuan192/data/Bee-Training-Data-Stage1/data",
        help="Path to local HF dataset (default: Bee-Training-Data-Stage1/data).",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train).")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save exported images (default: exported_images).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to export (default: 100).",
    )
    parser.add_argument(
        "--image-column",
        default="images",
        help='Image column name (default: "images").',
    )
    args = parser.parse_args()

    if args.num_images <= 0:
        raise ValueError("--num-images must be > 0")

    ds = load_local_dataset(args.dataset_path, args.split)
    image_col = pick_image_column(ds, args.image_column)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = min(args.num_images, len(ds))
    if total < args.num_images:
        print(f"Requested {args.num_images} images, but dataset has {len(ds)} rows; exporting {total}.")

    for i in range(total):
        example = ds[i]
        image_obj = example[image_col]
        out_path = output_dir / f"image_{i:05d}.png"
        save_example_image(image_obj, out_path)

    print(f"Exported {total} images from '{args.dataset_path}' to '{output_dir}'.")


if __name__ == "__main__":
    main()

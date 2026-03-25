"""
python compare_cknna_penguin_siglip2.py \
  --images data/ \
  --out results/penguin_vs_siglip2 \
  --max-images 256 \
  --batch-size 8 \
  --topk 10 \
  --drop-embedding-layer
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from metrics import AlignmentMetrics


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="data/", help="Folder of images")
    p.add_argument("--out", type=str, default="results/", help="Output directory")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    p.add_argument(
        "--drop-embedding-layer",
        action="store_true",
        help="Drop layer 0 (embedding output) from both models before scoring",
    )
    p.add_argument(
        "--from-last-layer",
        action="store_true",
        help="Compare layer pairs from last to first (instead of first to last)",
    )
    return p.parse_args()


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def list_images(folder: str, max_images: int, seed: int) -> list[Path]:
    paths = [
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    if not paths:
        raise ValueError(f"No images found under: {folder}")

    rng = random.Random(seed)
    rng.shuffle(paths)

    if max_images is not None and max_images > 0:
        paths = paths[:max_images]

    if len(paths) < 3:
        raise ValueError("Need at least 3 images for CKNNA (and topk >= 2).")
    return paths


def load_pil_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for p in paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return images


def mean_pool_penguin_hidden_state(hidden_state: torch.Tensor, token_counts: list[int]) -> torch.Tensor:
    """
    hidden_state: [1, total_tokens, hidden_dim]
    returns: [batch, hidden_dim]
    """
    chunks = hidden_state.squeeze(0).split(token_counts, dim=0)
    pooled = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
    return pooled


def extract_penguin_features(
    model,
    processor,
    images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [batch, num_layers, hidden_dim]
    """
    # IMPORTANT: nested list -> each image is its own clip
    batch = processor(
        images=[[img] for img in images],
        merge_size=1,
        return_tensors="pt",
    )

    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
    grid_sizes = batch["grid_sizes"].to(device)
    merge_sizes = batch["merge_sizes"].to(device)

    with torch.inference_mode():
        inputs_embeds = model.embeddings(pixel_values)
        encoder_out = model.encoder(
            inputs_embeds=inputs_embeds[None, ...],
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            output_hidden_states=True,
            use_cache=False,
        )

    token_counts = grid_sizes.prod(dim=1).tolist()

    per_layer = []
    for hs in encoder_out.hidden_states:
        pooled = mean_pool_penguin_hidden_state(hs, token_counts)
        per_layer.append(pooled.float().cpu())

    return torch.stack(per_layer, dim=1)  # [B, Lp, Dp]


def extract_siglip2_features(
    model,
    processor,
    images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [batch, num_layers, hidden_dim]
    """
    batch = processor(images=images, return_tensors="pt")
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    batch["pixel_values"] = batch["pixel_values"].to(dtype=dtype)

    with torch.inference_mode():
        vision_out = model.vision_model(
            pixel_values=batch["pixel_values"],
            # pixel_attention_mask=batch.get("pixel_attention_mask"),
            # spatial_shapes=batch.get("spatial_shapes"),
            output_hidden_states=True,
        )

    per_layer = []
    for hs in vision_out.hidden_states:
        # SigLIP2 uses a pooled first token; drop token 0 so we're averaging patch tokens.
        tokens = hs[:, 1:, :] if hs.shape[1] > 1 else hs
        pooled = tokens.mean(dim=1)
        per_layer.append(pooled.float().cpu())

    return torch.stack(per_layer, dim=1)  # [B, Ls, Ds]


def compute_cknna_layerwise(
    penguin_feats: torch.Tensor,  # [N, Lp, Dp]
    siglip_feats: torch.Tensor,   # [N, Ls, Ds]
    topk: int,
    from_last_layer: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = penguin_feats.shape[0]
    topk = min(topk, n - 1)
    if topk < 2:
        raise ValueError("Need topk >= 2 for CKNNA.")

    lp = penguin_feats.shape[1]
    ls = siglip_feats.shape[1]
    compared_layers = min(lp, ls)
    scores = np.zeros((compared_layers,), dtype=np.float32)

    if from_last_layer:
        penguin_layer_indices = np.arange(lp - 1, lp - compared_layers - 1, -1, dtype=np.int32)
        siglip2_layer_indices = np.arange(ls - 1, ls - compared_layers - 1, -1, dtype=np.int32)
    else:
        penguin_layer_indices = np.arange(compared_layers, dtype=np.int32)
        siglip2_layer_indices = np.arange(compared_layers, dtype=np.int32)

    for out_idx in tqdm(range(compared_layers), desc="CKNNA layerwise"):
        penguin_idx = int(penguin_layer_indices[out_idx])
        siglip2_idx = int(siglip2_layer_indices[out_idx])
        score = AlignmentMetrics.cknna(
            penguin_feats[:, penguin_idx, :],
            siglip_feats[:, siglip2_idx, :],
            topk=topk,
        )
        scores[out_idx] = float(score)

    return scores, penguin_layer_indices, siglip2_layer_indices


def save_layerwise_plot(
    scores: np.ndarray,
    x_layer_indices: np.ndarray,
    out_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Could not import matplotlib. Install it with `pip install matplotlib` "
            "to save the layerwise plot."
        )
        return False

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x_layer_indices, scores, marker="o", linewidth=1.8)
    ax.set_xlabel("Layer idx")
    ax.set_ylabel("Similarity score")
    ax.set_title("CKNNA: Penguin vs SigLIP2 (layer-by-layer)")
    ax.grid(True, alpha=0.3, linewidth=0.7)
    if x_layer_indices.shape[0] >= 2 and x_layer_indices[0] > x_layer_indices[-1]:
        ax.set_xlim(int(x_layer_indices[0]), int(x_layer_indices[-1]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    dtype = pick_dtype(device)

    print(f"Using device={device}, dtype={dtype}")

    image_paths = list_images(args.images, args.max_images, args.seed)
    print(f"Found {len(image_paths)} images")

    # Load models
    penguin = AutoModel.from_pretrained(
        "tencent/Penguin-Encoder",
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    penguin_processor = AutoImageProcessor.from_pretrained(
        "tencent/Penguin-Encoder",
        trust_remote_code=True,
    )

    siglip2 = AutoModel.from_pretrained(
        "google/siglip2-so400m-patch14-384",
        torch_dtype=dtype,
    ).to(device).eval()

    siglip2_processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384"
    )

    penguin_batches = []
    siglip2_batches = []

    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Feature batches"):
        batch_paths = image_paths[start:start + args.batch_size]
        images = load_pil_images(batch_paths)

        penguin_feats = extract_penguin_features(
            penguin, penguin_processor, images, device, dtype
        )
        siglip2_feats = extract_siglip2_features(
            siglip2, siglip2_processor, images, device, dtype
        )

        penguin_batches.append(penguin_feats)
        siglip2_batches.append(siglip2_feats)

    penguin_feats = torch.cat(penguin_batches, dim=0)   # [N, Lp, Dp]
    siglip2_feats = torch.cat(siglip2_batches, dim=0)   # [N, Ls, Ds]

    if args.drop_embedding_layer:
        penguin_feats = penguin_feats[:, 1:, :]
        siglip2_feats = siglip2_feats[:, 1:, :]

    # Normalize because metrics.py uses dot products
    penguin_feats = F.normalize(penguin_feats, dim=-1)
    siglip2_feats = F.normalize(siglip2_feats, dim=-1)

    effective_topk = min(args.topk, penguin_feats.shape[0] - 1)
    scores, penguin_layer_indices, siglip2_layer_indices = compute_cknna_layerwise(
        penguin_feats,
        siglip2_feats,
        effective_topk,
        from_last_layer=args.from_last_layer,
    )
    best_step = int(np.nanargmax(scores))

    plot_path = out_dir / "cknna_layerwise.png"
    plot_created = save_layerwise_plot(scores, penguin_layer_indices, plot_path)
    scores_path = out_dir / "cknna_layerwise.npy"

    summary = {
        "num_images": int(penguin_feats.shape[0]),
        "topk": int(effective_topk),
        "penguin_layers": int(penguin_feats.shape[1]),
        "siglip2_layers": int(siglip2_feats.shape[1]),
        "compared_layers": int(scores.shape[0]),
        "comparison_direction": "last_to_first" if args.from_last_layer else "first_to_last",
        "best_score": float(scores[best_step]),
        "best_step": int(best_step),
        "best_penguin_layer_idx": int(penguin_layer_indices[best_step]),
        "best_siglip2_layer_idx": int(siglip2_layer_indices[best_step]),
        "last_layer_score": float(scores[-1]),
        "score_type": "layer_by_layer_cknna",
        "drop_embedding_layer": bool(args.drop_embedding_layer),
        "from_last_layer": bool(args.from_last_layer),
        "plot_created": bool(plot_created),
        "plot_path": str(plot_path) if plot_created else None,
        "layer_scores_path": str(scores_path),
    }

    np.save(scores_path, scores)
    np.savetxt(
        out_dir / "cknna_layerwise.csv",
        np.column_stack(
            (
                np.arange(scores.shape[0], dtype=np.int32),
                penguin_layer_indices,
                siglip2_layer_indices,
                scores,
            )
        ),
        delimiter=",",
        header="step_idx,penguin_layer_idx,siglip2_layer_idx,similarity_score",
        comments="",
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "image_paths.txt", "w") as f:
        for p in image_paths:
            f.write(str(p) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved layerwise scores to: {scores_path}")
    print(f"Saved layerwise CSV to: {out_dir / 'cknna_layerwise.csv'}")
    if plot_created:
        print(f"Saved layerwise plot to: {plot_path}")
    print(f"Saved summary to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

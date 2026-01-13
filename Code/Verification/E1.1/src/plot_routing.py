import argparse
import importlib
from pathlib import Path

import numpy as np


def _maybe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Missing dependency '{name}'. Install it in conda env 'moe'.\n"
            f"Example: `pip install {name}`\n"
            f"Original error: {e}"
        ) from e


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _as_list_str(arr) -> list[str]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [str(x) for x in arr.tolist()]
    if isinstance(arr, np.ndarray) and arr.ndim == 1:
        return [str(x) for x in arr.tolist()]
    return [str(arr)]


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _prob_from_counts(counts: np.ndarray, eps: float = 0.0) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    s = float(x.sum())
    if s <= 0:
        return np.zeros_like(x)
    p = x / s
    if eps > 0:
        p = p + eps
        p = p / p.sum()
    return p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--top_experts", type=int, default=50)
    parser.add_argument(
        "--phase",
        default="total",
        choices=["total", "prefill", "decode"],
        help="Which phase to plot. If phase counters are missing, falls back to total.",
    )
    args = parser.parse_args()

    if args.npz:
        npz_path = Path(args.npz)
        domain_dir = npz_path.parent
    else:
        if not args.run_dir or not args.domain:
            raise RuntimeError("Provide either --npz or both --run_dir and --domain")
        domain_dir = Path(args.run_dir) / args.domain
        npz_path = domain_dir / "routing_stats_merged.npz"

    if not npz_path.exists():
        raise RuntimeError(f"Missing merged npz: {npz_path}")

    # Ensure plotting works on headless servers.
    mpl = _maybe_import("matplotlib")
    try:
        mpl.use("Agg")
    except Exception:
        pass
    plt = _maybe_import("matplotlib.pyplot")

    data = _load_npz(npz_path)
    layer_names = _as_list_str(data.get("layer_names"))
    k_experts = int(np.asarray(data["k_experts"]).item())

    phase = (args.phase or "total").lower()
    if phase == "total":
        cnt_top1 = np.asarray(data["cnt_top1"]).astype(np.int64)
        cnt_topk = np.asarray(data["cnt_topk"]).astype(np.int64)
        tokens = np.asarray(data["tokens_processed"]).astype(np.int64)
        cnt_shared = np.asarray(data["cnt_shared"]).astype(np.int64) if "cnt_shared" in data else None
        phase_suffix = ""
        phase_title = "total"
    else:
        suffix = f"_{phase}"
        if f"cnt_top1{suffix}" in data and f"cnt_topk{suffix}" in data and f"tokens{suffix}" in data:
            cnt_top1 = np.asarray(data[f"cnt_top1{suffix}"]).astype(np.int64)
            cnt_topk = np.asarray(data[f"cnt_topk{suffix}"]).astype(np.int64)
            tokens = np.asarray(data[f"tokens{suffix}"]).astype(np.int64)
            cnt_shared = (
                np.asarray(data[f"cnt_shared{suffix}"]).astype(np.int64)
                if f"cnt_shared{suffix}" in data
                else None
            )
            phase_suffix = suffix
            phase_title = phase
        else:
            # Fall back to total.
            cnt_top1 = np.asarray(data["cnt_top1"]).astype(np.int64)
            cnt_topk = np.asarray(data["cnt_topk"]).astype(np.int64)
            tokens = np.asarray(data["tokens_processed"]).astype(np.int64)
            cnt_shared = np.asarray(data["cnt_shared"]).astype(np.int64) if "cnt_shared" in data else None
            phase_suffix = ""
            phase_title = f"{phase} (fallback=total)"

    num_layers, num_experts = cnt_top1.shape
    log_e = np.log(float(num_experts))

    # Per-layer metrics.
    top1_entropy_norm = np.zeros(num_layers, dtype=np.float64)
    topk_entropy_norm = np.zeros(num_layers, dtype=np.float64)
    top1_max_share = np.zeros(num_layers, dtype=np.float64)
    topk_max_share = np.zeros(num_layers, dtype=np.float64)
    shared_frac = np.zeros(num_layers, dtype=np.float64)

    for l in range(num_layers):
        n = int(tokens[l])
        if n <= 0:
            continue
        p1 = _prob_from_counts(cnt_top1[l])
        pk = _prob_from_counts(cnt_topk[l])
        top1_entropy_norm[l] = _entropy(p1) / log_e if log_e > 0 else 0.0
        topk_entropy_norm[l] = _entropy(pk) / log_e if log_e > 0 else 0.0
        top1_max_share[l] = float(p1.max()) if p1.size else 0.0
        topk_max_share[l] = float(pk.max()) if pk.size else 0.0
        shared_frac[l] = float((cnt_shared[l] / n) if cnt_shared is not None else 0.0)

    plots_dir = domain_dir / "plots"
    _ensure_dir(plots_dir)

    xs = np.arange(num_layers)

    # 1) Entropy curves.
    plt.figure(figsize=(10, 4))
    plt.plot(xs, top1_entropy_norm, label="top1 entropy / logE")
    plt.plot(xs, topk_entropy_norm, label=f"topk entropy / logE (k={k_experts})")
    plt.xlabel("Layer")
    plt.ylabel("Normalized entropy")
    plt.title(f"Routing entropy by layer ({phase_title})")
    plt.grid(True, alpha=0.2)
    plt.legend()
    out1 = plots_dir / f"entropy_by_layer{phase_suffix}.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()

    # 2) Max share curves.
    plt.figure(figsize=(10, 4))
    plt.plot(xs, top1_max_share, label="top1 max share")
    plt.plot(xs, topk_max_share, label="topk max share")
    plt.xlabel("Layer")
    plt.ylabel("Max expert probability")
    plt.title(f"Max expert share by layer ({phase_title})")
    plt.grid(True, alpha=0.2)
    plt.legend()
    out2 = plots_dir / f"max_share_by_layer{phase_suffix}.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()

    # 3) Shared fraction.
    plt.figure(figsize=(10, 4))
    plt.plot(xs, shared_frac, label="shared frac")
    plt.xlabel("Layer")
    plt.ylabel("Shared tokens / tokens_processed")
    plt.title(f"Shared expert fraction by layer ({phase_title})")
    plt.grid(True, alpha=0.2)
    out3 = plots_dir / f"shared_frac_by_layer{phase_suffix}.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=160)
    plt.close()

    # 4) Heatmap over top-N experts by global top1 usage.
    top_n = int(args.top_experts)
    top_n = max(1, min(top_n, num_experts))

    global_counts = cnt_top1.sum(axis=0)
    top_idx = np.argsort(-global_counts)[:top_n]
    mat = cnt_top1[:, top_idx].astype(np.float64)
    # normalize rows to probabilities for visualization
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    plt.figure(figsize=(max(8, top_n * 0.18), 6))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="P(expert|layer) (top1, restricted to top-N)")
    plt.xlabel(f"Top-{top_n} experts (by global top1 count)")
    plt.ylabel("Layer")
    plt.title(f"Top-1 routing distribution (top experts; {phase_title})")
    out4 = plots_dir / f"heatmap_top{top_n}_experts_top1{phase_suffix}.png"
    plt.tight_layout()
    plt.savefig(out4, dpi=160)
    plt.close()

    print(f"[E1.1] Wrote plots under: {plots_dir}")
    print(f"  - {out1.name}")
    print(f"  - {out2.name}")
    print(f"  - {out3.name}")
    print(f"  - {out4.name}")

    # 5) Heatmap over top-N experts by global topk usage.
    # This visualizes overall load (top-k membership), not just the top-1 winner.
    global_counts_k = cnt_topk.sum(axis=0)
    top_idx_k = np.argsort(-global_counts_k)[:top_n]
    mat_k = cnt_topk[:, top_idx_k].astype(np.float64)
    # normalize rows to probabilities for visualization
    row_sums_k = mat_k.sum(axis=1, keepdims=True)
    row_sums_k[row_sums_k == 0] = 1.0
    mat_k = mat_k / row_sums_k

    plt.figure(figsize=(max(8, top_n * 0.18), 6))
    plt.imshow(mat_k, aspect="auto", interpolation="nearest")
    plt.colorbar(label=f"P(expert|layer) (topk, restricted to top-N; k={k_experts})")
    plt.xlabel(f"Top-{top_n} experts (by global topk count)")
    plt.ylabel("Layer")
    plt.title(f"Top-k routing distribution (top experts; {phase_title})")
    out5 = plots_dir / f"heatmap_top{top_n}_experts_topk{phase_suffix}.png"
    plt.tight_layout()
    plt.savefig(out5, dpi=160)
    plt.close()

    print(f"  - {out5.name}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def _safe_prob_from_counts(counts: np.ndarray, total: int, eps: float = 0.0) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    p = counts / float(total)
    if eps > 0:
        p = p + eps
        p = p / p.sum()
    return p


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _gini_from_nonneg(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1, dtype=np.float64)
    # Gini = (2*sum(i*xi))/(n*sum(x)) - (n+1)/n
    return float((2.0 * (i * xs).sum()) / (n * s) - (n + 1.0) / n)


def _cv(p: np.ndarray) -> float:
    # coefficient of variation over probabilities
    m = float(p.mean())
    if m <= 0:
        return 0.0
    return float(p.std() / m)


def summarize_npz(npz_path: Path) -> dict:
    data = _load_npz(npz_path)

    layer_names = _as_list_str(data.get("layer_names"))
    k_experts = int(np.asarray(data["k_experts"]).item())

    # Backward compatible: default to total counters.
    cnt_top1 = np.asarray(data["cnt_top1"]).astype(np.int64)
    cnt_topk = np.asarray(data["cnt_topk"]).astype(np.int64)
    tokens_processed = np.asarray(data["tokens_processed"]).astype(np.int64)
    cnt_shared = None
    if "cnt_shared" in data:
        cnt_shared = np.asarray(data["cnt_shared"]).astype(np.int64)

    if cnt_top1.ndim != 2:
        raise RuntimeError(f"Expected cnt_top1 to be 2D [L,E], got shape={cnt_top1.shape}")

    num_layers, num_experts = cnt_top1.shape
    log_e = np.log(float(num_experts))

    layer_rows: list[dict] = []
    for l in range(num_layers):
        n = int(tokens_processed[l])
        if n <= 0:
            continue

        p1 = _safe_prob_from_counts(cnt_top1[l], n)
        pk = _safe_prob_from_counts(cnt_topk[l], n * k_experts)

        h1 = _entropy(p1)
        hk = _entropy(pk)

        layer_rows.append(
            {
                "layer": l,
                "layer_name": layer_names[l] if l < len(layer_names) else f"layer_{l}",
                "tokens": n,
                "k_experts": k_experts,
                "num_experts": num_experts,
                "shared_tokens": int(cnt_shared[l]) if cnt_shared is not None else 0,
                "shared_frac": float((cnt_shared[l] / n) if cnt_shared is not None else 0.0),
                "top1_entropy": float(h1),
                "top1_entropy_norm": float(h1 / log_e) if log_e > 0 else 0.0,
                "top1_eff_experts": float(np.exp(h1)),
                "top1_max_share": float(p1.max()),
                "top1_cv": float(_cv(p1)),
                "top1_gini": float(_gini_from_nonneg(p1)),
                "topk_entropy": float(hk),
                "topk_entropy_norm": float(hk / log_e) if log_e > 0 else 0.0,
                "topk_eff_experts": float(np.exp(hk)),
                "topk_max_share": float(pk.max()),
                "topk_cv": float(_cv(pk)),
                "topk_gini": float(_gini_from_nonneg(pk)),
            }
        )

    if not layer_rows:
        raise RuntimeError(f"No valid layers found (tokens_processed all <=0?) in {npz_path}")

    # Token-weighted global summaries.
    tokens = np.array([r["tokens"] for r in layer_rows], dtype=np.float64)
    w = tokens / tokens.sum()

    def _wavg(key: str) -> float:
        return float(np.sum(w * np.array([r[key] for r in layer_rows], dtype=np.float64)))

    summary = {
        "npz": str(npz_path),
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "k_experts": int(k_experts),
        "total_tokens": int(tokens.sum()),
        "metrics_token_weighted": {
            "shared_frac": _wavg("shared_frac"),
            "top1_entropy_norm": _wavg("top1_entropy_norm"),
            "top1_gini": _wavg("top1_gini"),
            "top1_cv": _wavg("top1_cv"),
            "top1_max_share": _wavg("top1_max_share"),
            "topk_entropy_norm": _wavg("topk_entropy_norm"),
            "topk_gini": _wavg("topk_gini"),
            "topk_cv": _wavg("topk_cv"),
            "topk_max_share": _wavg("topk_max_share"),
        },
        "layers": layer_rows,
    }

    return summary


def summarize_npz_phase(npz_path: Path, phase: str) -> dict:
    """Summarize a specific phase: total/prefill/decode.

    If phase-specific counters are missing, falls back to total.
    """
    data = _load_npz(npz_path)
    phase = (phase or "total").lower()
    if phase not in {"total", "prefill", "decode"}:
        raise RuntimeError(f"Unknown phase: {phase}")

    if phase == "total":
        return summarize_npz(npz_path)

    suffix = f"_{phase}"
    # Fall back to total if missing.
    if f"cnt_top1{suffix}" not in data or f"cnt_topk{suffix}" not in data or f"tokens{suffix}" not in data:
        out = summarize_npz(npz_path)
        out["phase"] = phase
        out["phase_fallback_to_total"] = True
        return out

    # Build a temporary dict with total-like keys and reuse summarize_npz.
    tmp = dict(data)
    tmp["cnt_top1"] = tmp[f"cnt_top1{suffix}"]
    tmp["cnt_topk"] = tmp[f"cnt_topk{suffix}"]
    tmp["tokens_processed"] = tmp[f"tokens{suffix}"]
    if f"cnt_shared{suffix}" in tmp:
        tmp["cnt_shared"] = tmp[f"cnt_shared{suffix}"]
    elif "cnt_shared" in tmp:
        # If shared exists only as total, do not reuse it for phase.
        tmp.pop("cnt_shared", None)

    # Write to a temp npz in-memory is not possible; reimplement minimal load.
    layer_names = _as_list_str(tmp.get("layer_names"))
    cnt_top1 = np.asarray(tmp["cnt_top1"]).astype(np.int64)
    cnt_topk = np.asarray(tmp["cnt_topk"]).astype(np.int64)
    tokens_processed = np.asarray(tmp["tokens_processed"]).astype(np.int64)
    k_experts = int(np.asarray(tmp["k_experts"]).item())
    cnt_shared = None
    if "cnt_shared" in tmp:
        cnt_shared = np.asarray(tmp["cnt_shared"]).astype(np.int64)

    if cnt_top1.ndim != 2:
        raise RuntimeError(f"Expected cnt_top1 to be 2D [L,E], got shape={cnt_top1.shape}")

    num_layers, num_experts = cnt_top1.shape
    log_e = np.log(float(num_experts))

    layer_rows: list[dict] = []
    for l in range(num_layers):
        n = int(tokens_processed[l])
        if n <= 0:
            continue

        p1 = _safe_prob_from_counts(cnt_top1[l], n)
        pk = _safe_prob_from_counts(cnt_topk[l], n * k_experts)

        h1 = _entropy(p1)
        hk = _entropy(pk)

        layer_rows.append(
            {
                "layer": l,
                "layer_name": layer_names[l] if l < len(layer_names) else f"layer_{l}",
                "tokens": n,
                "k_experts": k_experts,
                "num_experts": num_experts,
                "shared_tokens": int(cnt_shared[l]) if cnt_shared is not None else 0,
                "shared_frac": float((cnt_shared[l] / n) if cnt_shared is not None else 0.0),
                "top1_entropy": float(h1),
                "top1_entropy_norm": float(h1 / log_e) if log_e > 0 else 0.0,
                "top1_eff_experts": float(np.exp(h1)),
                "top1_max_share": float(p1.max()),
                "top1_cv": float(_cv(p1)),
                "top1_gini": float(_gini_from_nonneg(p1)),
                "topk_entropy": float(hk),
                "topk_entropy_norm": float(hk / log_e) if log_e > 0 else 0.0,
                "topk_eff_experts": float(np.exp(hk)),
                "topk_max_share": float(pk.max()),
                "topk_cv": float(_cv(pk)),
                "topk_gini": float(_gini_from_nonneg(pk)),
            }
        )

    if not layer_rows:
        raise RuntimeError(f"No valid layers found (tokens_processed all <=0?) in {npz_path}")

    tokens = np.array([r["tokens"] for r in layer_rows], dtype=np.float64)
    w = tokens / tokens.sum()

    def _wavg(key: str) -> float:
        return float(np.sum(w * np.array([r[key] for r in layer_rows], dtype=np.float64)))

    summary = {
        "npz": str(npz_path),
        "phase": phase,
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "k_experts": int(k_experts),
        "total_tokens": int(tokens.sum()),
        "metrics_token_weighted": {
            "shared_frac": _wavg("shared_frac"),
            "top1_entropy_norm": _wavg("top1_entropy_norm"),
            "top1_gini": _wavg("top1_gini"),
            "top1_cv": _wavg("top1_cv"),
            "top1_max_share": _wavg("top1_max_share"),
            "topk_entropy_norm": _wavg("topk_entropy_norm"),
            "topk_gini": _wavg("topk_gini"),
            "topk_cv": _wavg("topk_cv"),
            "topk_max_share": _wavg("topk_max_share"),
        },
        "layers": layer_rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default=None,
        help="Path to routing_stats_merged.npz. If omitted, use --run_dir/--domain.",
    )
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument(
        "--phase",
        default="total",
        choices=["total", "prefill", "decode"],
        help="Which phase to summarize. If phase counters are missing, falls back to total.",
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

    summary = summarize_npz_phase(npz_path, args.phase)

    suffix = "" if args.phase == "total" else f"_{args.phase}"
    out_json = domain_dir / f"routing_summary{suffix}.json"
    out_csv = domain_dir / f"routing_summary_layers{suffix}.csv"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Flat layer metrics CSV.
    fieldnames = [
        "layer",
        "layer_name",
        "tokens",
        "k_experts",
        "num_experts",
        "shared_tokens",
        "shared_frac",
        "top1_entropy_norm",
        "top1_eff_experts",
        "top1_max_share",
        "top1_cv",
        "top1_gini",
        "topk_entropy_norm",
        "topk_eff_experts",
        "topk_max_share",
        "topk_cv",
        "topk_gini",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary["layers"]:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"[E1.1] Wrote: {out_json}")
    print(f"[E1.1] Wrote: {out_csv}")
    print("[E1.1] Token-weighted metrics:")
    for k, v in summary["metrics_token_weighted"].items():
        print(f"  - {k}: {v:.6f}")


if __name__ == "__main__":
    main()

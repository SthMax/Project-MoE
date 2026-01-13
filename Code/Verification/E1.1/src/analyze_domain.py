import argparse
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
    return float((2.0 * (i * xs).sum()) / (n * s) - (n + 1.0) / n)


def _cv(p: np.ndarray) -> float:
    m = float(p.mean())
    if m <= 0:
        return 0.0
    return float(p.std() / m)


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


def _top_experts_from_counts(counts_1d: np.ndarray, top_n: int) -> list[dict]:
    counts_1d = np.asarray(counts_1d, dtype=np.int64)
    total = int(counts_1d.sum())
    if total <= 0:
        return []
    top_n = max(1, min(int(top_n), counts_1d.size))
    idx = np.argsort(-counts_1d)[:top_n]
    out = []
    for rank, e in enumerate(idx.tolist(), start=1):
        c = int(counts_1d[e])
        out.append(
            {
                "rank": rank,
                "expert": int(e),
                "count": c,
                "share": float(c / total),
            }
        )
    return out


def _analyze_domain_phase(data: dict, npz_path: Path, phase: str, top_n: int) -> dict:
    phase = (phase or "total").lower()
    if phase not in {"total", "prefill", "decode"}:
        raise RuntimeError(f"Unknown phase: {phase}")

    layer_names = _as_list_str(data.get("layer_names"))
    k_experts = int(np.asarray(data["k_experts"]).item())

    if phase == "total":
        cnt_top1 = np.asarray(data["cnt_top1"]).astype(np.int64)
        cnt_topk = np.asarray(data["cnt_topk"]).astype(np.int64)
        tokens = np.asarray(data["tokens_processed"]).astype(np.int64)
        cnt_shared = np.asarray(data["cnt_shared"]).astype(np.int64) if "cnt_shared" in data else None
        fallback = False
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
            fallback = False
        else:
            cnt_top1 = np.asarray(data["cnt_top1"]).astype(np.int64)
            cnt_topk = np.asarray(data["cnt_topk"]).astype(np.int64)
            tokens = np.asarray(data["tokens_processed"]).astype(np.int64)
            cnt_shared = np.asarray(data["cnt_shared"]).astype(np.int64) if "cnt_shared" in data else None
            fallback = True

    num_layers, num_experts = cnt_top1.shape
    log_e = np.log(float(num_experts))

    layers = []
    for l in range(num_layers):
        n = int(tokens[l])
        if n <= 0:
            continue

        p1 = _prob_from_counts(cnt_top1[l])
        pk = _prob_from_counts(cnt_topk[l])

        h1 = _entropy(p1)
        hk = _entropy(pk)

        layers.append(
            {
                "layer": int(l),
                "layer_name": layer_names[l] if l < len(layer_names) else f"layer_{l}",
                "tokens": n,
                "k_experts": int(k_experts),
                "num_experts": int(num_experts),
                "shared_tokens": int(cnt_shared[l]) if cnt_shared is not None else 0,
                "shared_frac": float((cnt_shared[l] / n) if cnt_shared is not None else 0.0),
                "top1": {
                    "entropy": float(h1),
                    "entropy_norm": float(h1 / log_e) if log_e > 0 else 0.0,
                    "eff_experts": float(np.exp(h1)),
                    "max_share": float(p1.max()) if p1.size else 0.0,
                    "cv": float(_cv(p1)),
                    "gini": float(_gini_from_nonneg(p1)),
                    "top_experts": _top_experts_from_counts(cnt_top1[l], top_n=top_n),
                },
                "topk": {
                    "entropy": float(hk),
                    "entropy_norm": float(hk / log_e) if log_e > 0 else 0.0,
                    "eff_experts": float(np.exp(hk)),
                    "max_share": float(pk.max()) if pk.size else 0.0,
                    "cv": float(_cv(pk)),
                    "gini": float(_gini_from_nonneg(pk)),
                    "top_experts": _top_experts_from_counts(cnt_topk[l], top_n=top_n),
                },
            }
        )

    top1_max = np.array([x["top1"]["max_share"] for x in layers], dtype=np.float64)
    top1_ent = np.array([x["top1"]["entropy_norm"] for x in layers], dtype=np.float64)

    def _robust_thresh_high(x: np.ndarray, k: float = 3.0) -> float:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        if mad <= 0:
            return float(np.max(x))
        return med + k * 1.4826 * mad

    def _robust_thresh_low(x: np.ndarray, k: float = 3.0) -> float:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        if mad <= 0:
            return float(np.min(x))
        return med - k * 1.4826 * mad

    thr_max_high = _robust_thresh_high(top1_max, k=3.0) if layers else 0.0
    thr_ent_low = _robust_thresh_low(top1_ent, k=3.0) if layers else 0.0

    flagged = []
    for x in layers:
        reasons = []
        if x["top1"]["max_share"] >= thr_max_high:
            reasons.append("top1_max_share_high")
        if x["top1"]["entropy_norm"] <= thr_ent_low:
            reasons.append("top1_entropy_norm_low")
        if reasons:
            flagged.append({"layer": x["layer"], "layer_name": x["layer_name"], "reasons": reasons})

    return {
        "phase": phase,
        "phase_fallback_to_total": bool(fallback),
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "k_experts": int(k_experts),
        "top_n": int(top_n),
        "flag_thresholds": {
            "top1_max_share_high": float(thr_max_high),
            "top1_entropy_norm_low": float(thr_ent_low),
        },
        "flagged_layers": flagged,
        "layers": layers,
    }


def analyze_domain(npz_path: Path, top_n: int = 50) -> dict:
    data = _load_npz(npz_path)
    return {
        "npz": str(npz_path),
        "analysis": {
            "total": _analyze_domain_phase(data, npz_path, "total", top_n=top_n),
            "prefill": _analyze_domain_phase(data, npz_path, "prefill", top_n=top_n),
            "decode": _analyze_domain_phase(data, npz_path, "decode", top_n=top_n),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--top_experts", type=int, default=50)
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

    out = analyze_domain(npz_path, top_n=int(args.top_experts))
    out_path = domain_dir / "routing_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E1.1] Wrote: {out_path}")


if __name__ == "__main__":
    main()

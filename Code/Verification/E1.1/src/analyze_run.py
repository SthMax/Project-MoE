import argparse
import json
from itertools import combinations
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


def _prob(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    s = float(x.sum())
    if s <= 0:
        return np.zeros_like(x)
    p = x / s
    p = p + eps
    p = p / p.sum()
    return p


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence (natural log)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(p) - np.log(m))).sum())
    kl_qm = float((q * (np.log(q) - np.log(m))).sum())
    return 0.5 * (kl_pm + kl_qm)


def _load_domain_merged(run_dir: Path, domain: str) -> dict:
    domain_dir = run_dir / domain
    npz_path = domain_dir / "routing_stats_merged.npz"
    if not npz_path.exists():
        raise RuntimeError(f"Missing merged npz for domain={domain}: {npz_path}")
    data = _load_npz(npz_path)
    layer_names = _as_list_str(data.get("layer_names"))
    cnt_top1 = np.asarray(data["cnt_top1"]).astype(np.int64)
    cnt_topk = np.asarray(data["cnt_topk"]).astype(np.int64)
    tokens_processed = np.asarray(data["tokens_processed"]).astype(np.int64)
    k_experts = int(np.asarray(data["k_experts"]).item())

    # Optional phase-split.
    cnt_top1_prefill = (
        np.asarray(data["cnt_top1_prefill"]).astype(np.int64) if "cnt_top1_prefill" in data else None
    )
    cnt_topk_prefill = (
        np.asarray(data["cnt_topk_prefill"]).astype(np.int64) if "cnt_topk_prefill" in data else None
    )
    cnt_top1_decode = (
        np.asarray(data["cnt_top1_decode"]).astype(np.int64) if "cnt_top1_decode" in data else None
    )
    cnt_topk_decode = (
        np.asarray(data["cnt_topk_decode"]).astype(np.int64) if "cnt_topk_decode" in data else None
    )
    return {
        "domain": domain,
        "npz": str(npz_path),
        "layer_names": layer_names,
        "cnt_top1": cnt_top1,
        "cnt_topk": cnt_topk,
        "tokens_processed": tokens_processed,
        "cnt_top1_prefill": cnt_top1_prefill,
        "cnt_topk_prefill": cnt_topk_prefill,
        "cnt_top1_decode": cnt_top1_decode,
        "cnt_topk_decode": cnt_topk_decode,
        "k_experts": k_experts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help=".../outputs/runs/<run_id>")
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain list. If omitted, auto-detect domains with routing_stats_merged.npz.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise RuntimeError(f"run_dir not found: {run_dir}")

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    else:
        domains = []
        for p in run_dir.iterdir():
            if not p.is_dir():
                continue
            if (p / "routing_stats_merged.npz").exists():
                domains.append(p.name)
        domains = sorted(domains)

    if len(domains) < 1:
        raise RuntimeError(f"No domains with routing_stats_merged.npz under: {run_dir}")

    dom_data = {d: _load_domain_merged(run_dir, d) for d in domains}

    # Basic validation: layer count must match.
    layer_counts = {d: dom_data[d]["cnt_top1"].shape[0] for d in domains}
    if len(set(layer_counts.values())) != 1:
        raise RuntimeError(f"Layer count mismatch across domains: {layer_counts}")

    num_layers = next(iter(layer_counts.values()))

    comparisons = []
    for a, b in combinations(domains, 2):
        da = dom_data[a]
        db = dom_data[b]

        def _js_for(cnt1_a, cnt1_b, cntk_a, cntk_b):
            js1 = []
            jsk = []
            for l in range(num_layers):
                pa = _prob(cnt1_a[l])
                pb = _prob(cnt1_b[l])
                qa = _prob(cntk_a[l])
                qb = _prob(cntk_b[l])
                js1.append(_js_divergence(pa, pb))
                jsk.append(_js_divergence(qa, qb))
            js1 = np.asarray(js1, dtype=np.float64)
            jsk = np.asarray(jsk, dtype=np.float64)
            return {
                "js_top1_mean": float(js1.mean()),
                "js_top1_p50": float(np.quantile(js1, 0.50)),
                "js_top1_p90": float(np.quantile(js1, 0.90)),
                "js_topk_mean": float(jsk.mean()),
                "js_topk_p50": float(np.quantile(jsk, 0.50)),
                "js_topk_p90": float(np.quantile(jsk, 0.90)),
            }

        entry = {"pair": [a, b], "total": _js_for(da["cnt_top1"], db["cnt_top1"], da["cnt_topk"], db["cnt_topk"])}

        # Optional phases.
        if da.get("cnt_top1_prefill") is not None and db.get("cnt_top1_prefill") is not None:
            entry["prefill"] = _js_for(
                da["cnt_top1_prefill"],
                db["cnt_top1_prefill"],
                da["cnt_topk_prefill"],
                db["cnt_topk_prefill"],
            )
        if da.get("cnt_top1_decode") is not None and db.get("cnt_top1_decode") is not None:
            entry["decode"] = _js_for(
                da["cnt_top1_decode"],
                db["cnt_top1_decode"],
                da["cnt_topk_decode"],
                db["cnt_topk_decode"],
            )

        comparisons.append(entry)

    # Pull in per-domain summaries if present.
    per_domain = {}
    for d in domains:
        dom_dir = run_dir / d
        s_path = dom_dir / "routing_summary.json"
        if s_path.exists():
            per_domain[d] = json.loads(s_path.read_text(encoding="utf-8"))
        else:
            per_domain[d] = {"npz": dom_data[d]["npz"]}

    out = {
        "run_dir": str(run_dir),
        "domains": domains,
        "comparisons": comparisons,
        "per_domain": per_domain,
    }

    out_path = run_dir / "run_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E1.1] Wrote: {out_path}")


if __name__ == "__main__":
    main()

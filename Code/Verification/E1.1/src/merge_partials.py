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


def _is_counter_matrix(name: str, arr: np.ndarray) -> bool:
    return name.startswith("cnt_") and isinstance(arr, np.ndarray) and arr.ndim == 2


def _is_counter_vector(name: str, arr: np.ndarray) -> bool:
    return (
        (name.startswith("tokens_") or name.startswith("cnt_shared"))
        and isinstance(arr, np.ndarray)
        and arr.ndim == 1
    )


def _is_debug_vector(name: str, arr: np.ndarray) -> bool:
    return name.startswith("dbg_") and isinstance(arr, np.ndarray) and arr.ndim == 1


def _is_scalar_meta(name: str, arr: np.ndarray) -> bool:
    return name in {"k_experts", "num_experts"} and isinstance(arr, np.ndarray) and arr.ndim == 0


def _is_debug_scalar(name: str, arr: np.ndarray) -> bool:
    return name.startswith("dbg_") and isinstance(arr, np.ndarray) and arr.ndim == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help=".../outputs/runs/<run_id>")
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if partials have mismatched layer_names. "
            "By default, we auto-select the best-matching group of partials "
            "(useful when the partials dir contains leftovers from failed runs)."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    domain_dir = run_dir / args.domain
    partials_dir = domain_dir / "partials"

    partial_files = sorted(partials_dir.glob("*.npz"))
    if not partial_files:
        raise RuntimeError(f"No partial npz files found in {partials_dir}")

    # Load all partials and group by exact layer_names to handle leftovers from
    # previous failed runs.
    groups: dict[tuple[str, ...], list[tuple[Path, dict]]] = {}
    for p in partial_files:
        data = _load_npz(p)
        key = tuple(_as_list_str(data.get("layer_names")))
        groups.setdefault(key, []).append((p, data))

    if args.strict and len(groups) != 1:
        keys_preview = [list(k[:3]) for k in groups.keys()]
        raise RuntimeError(
            "Layer name mismatch between partials (strict mode).\n"
            f"Found {len(groups)} groups of layer_names. Examples: {keys_preview}"
        )

    # Choose the best group:
    # 1) most layers (prefer full model run)
    # 2) most total tokens_processed (prefer the most complete run)
    def _group_score(items: list[tuple[Path, dict]]) -> tuple[int, int]:
        example_data = items[0][1]
        layer_count = len(_as_list_str(example_data.get("layer_names")))
        total_tokens = 0
        for _, d in items:
            tp = d.get("tokens_processed")
            if tp is not None:
                total_tokens += int(np.asarray(tp).astype(np.int64).sum())
        return (layer_count, total_tokens)

    selected_key = max(groups.items(), key=lambda kv: _group_score(kv[1]))[0]
    selected_items = groups[selected_key]

    ignored_files = [p.name for p in partial_files if p not in {x[0] for x in selected_items}]
    if ignored_files:
        print(
            f"[E1.1] Warning: ignoring {len(ignored_files)} partial(s) with mismatched layer_names: "
            + ", ".join(ignored_files[:5])
            + (" ..." if len(ignored_files) > 5 else "")
        )

    merged_layer_names: list[str] | None = None
    merged: dict[str, np.ndarray] = {}
    scalar_meta: dict[str, int] = {}

    for p, data in selected_items:
        layer_names = _as_list_str(data.get("layer_names"))

        if merged_layer_names is None:
            merged_layer_names = layer_names
        else:
            if layer_names != merged_layer_names:
                raise RuntimeError(
                    f"Internal error: selected partials still mismatch layer_names. File={p.name}"
                )

        # Merge scalar meta.
        for key in ("k_experts", "num_experts"):
            if key in data:
                v = int(np.asarray(data[key]).item())
                if key in scalar_meta and scalar_meta[key] != v:
                    raise RuntimeError(f"{key} mismatch: {scalar_meta[key]} vs {v} (file={p.name})")
                scalar_meta[key] = v

        # Merge all counter arrays we recognize.
        for key, val in data.items():
            if key == "layer_names":
                continue
            if not isinstance(val, np.ndarray):
                continue

            if (
                _is_counter_matrix(key, val)
                or _is_counter_vector(key, val)
                or _is_debug_vector(key, val)
                or _is_scalar_meta(key, val)
                or _is_debug_scalar(key, val)
            ):
                arr = val
                # Normalize dtypes.
                if arr.dtype.kind in {"i", "u"}:
                    arr = arr.astype(np.int64, copy=False)
                elif arr.dtype.kind == "f":
                    arr = arr.astype(np.float64, copy=False)

                if key in merged:
                    if merged[key].shape != arr.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {key}: {merged[key].shape} vs {arr.shape} (file={p.name})"
                        )
                    merged[key] += arr
                else:
                    merged[key] = arr.copy()

    assert merged_layer_names is not None

    # Backward-compat: if phase-split exists but total is missing, synthesize totals.
    if "cnt_top1" not in merged and "cnt_top1_prefill" in merged and "cnt_top1_decode" in merged:
        merged["cnt_top1"] = merged["cnt_top1_prefill"] + merged["cnt_top1_decode"]
    if "cnt_topk" not in merged and "cnt_topk_prefill" in merged and "cnt_topk_decode" in merged:
        merged["cnt_topk"] = merged["cnt_topk_prefill"] + merged["cnt_topk_decode"]
    if "tokens_processed" not in merged and "tokens_prefill" in merged and "tokens_decode" in merged:
        merged["tokens_processed"] = merged["tokens_prefill"] + merged["tokens_decode"]
    if "cnt_shared" not in merged and "cnt_shared_prefill" in merged and "cnt_shared_decode" in merged:
        merged["cnt_shared"] = merged["cnt_shared_prefill"] + merged["cnt_shared_decode"]

    out_path = domain_dir / "routing_stats_merged.npz"
    np.savez_compressed(
        out_path,
        layer_names=np.array(merged_layer_names, dtype=object),
        **{k: v for k, v in merged.items() if k not in {"k_experts", "num_experts"}},
        **{k: np.int64(v) for k, v in scalar_meta.items()},
    )

    print(f"[E1.1] Merged {len(selected_items)} partials -> {out_path}")

    # Sanity checks (total + optional phase split).
    ok_topk = True
    ok_top1 = True
    k_experts = int(scalar_meta.get("k_experts", -1))

    def _check(prefix: str) -> tuple[bool, bool]:
        cnt_top1_key = f"cnt_top1{prefix}"
        cnt_topk_key = f"cnt_topk{prefix}"
        tokens_key = f"tokens{prefix}" if prefix else "tokens_processed"
        if cnt_top1_key not in merged or cnt_topk_key not in merged or tokens_key not in merged:
            return True, True
        cnt_top1 = merged[cnt_top1_key]
        cnt_topk = merged[cnt_topk_key]
        tokens = merged[tokens_key]
        sum_top1 = cnt_top1.sum(axis=1)
        sum_topk = cnt_topk.sum(axis=1)
        ok1 = np.all(sum_top1 == tokens)
        okk = True
        if k_experts > 0:
            okk = np.all(sum_topk == tokens * k_experts)
        return bool(okk), bool(ok1)

    ok_topk, ok_top1 = _check("")
    ok_topk_p, ok_top1_p = _check("_prefill")
    ok_topk_d, ok_top1_d = _check("_decode")

    print(f"[E1.1] Check total topk sum == tokens*k: {ok_topk}")
    print(f"[E1.1] Check total top1 sum == tokens: {ok_top1}")
    if "cnt_top1_prefill" in merged:
        print(f"[E1.1] Check prefill topk sum == tokens*k: {ok_topk_p}")
        print(f"[E1.1] Check prefill top1 sum == tokens: {ok_top1_p}")
    if "cnt_top1_decode" in merged:
        print(f"[E1.1] Check decode topk sum == tokens*k: {ok_topk_d}")
        print(f"[E1.1] Check decode top1 sum == tokens: {ok_top1_d}")

    meta_path = domain_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["merged_npz"] = str(out_path)
        meta["partials_count"] = len(selected_items)
        meta["ignored_partials_count"] = len(partial_files) - len(selected_items)
        meta["sanity_ok_topk"] = bool(ok_topk)
        meta["sanity_ok_top1"] = bool(ok_top1)
        if "cnt_top1_prefill" in merged:
            meta["sanity_ok_topk_prefill"] = bool(ok_topk_p)
            meta["sanity_ok_top1_prefill"] = bool(ok_top1_p)
        if "cnt_top1_decode" in merged:
            meta["sanity_ok_topk_decode"] = bool(ok_topk_d)
            meta["sanity_ok_top1_decode"] = bool(ok_top1_d)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

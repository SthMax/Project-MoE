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
    merged_cnt_topk = None
    merged_cnt_top1 = None
    merged_cnt_shared = None
    merged_tokens_processed = None
    k_experts = None

    for p, data in selected_items:
        layer_names = _as_list_str(data.get("layer_names"))

        cnt_topk = data["cnt_topk"].astype(np.int64)
        cnt_top1 = data["cnt_top1"].astype(np.int64)
        cnt_shared = data["cnt_shared"].astype(np.int64)
        tokens_processed = data["tokens_processed"].astype(np.int64)
        k = int(data.get("k_experts", -1))

        if merged_layer_names is None:
            merged_layer_names = layer_names
            merged_cnt_topk = cnt_topk
            merged_cnt_top1 = cnt_top1
            merged_cnt_shared = cnt_shared
            merged_tokens_processed = tokens_processed
            k_experts = k
        else:
            # Within a selected group, layer_names must match.
            if layer_names != merged_layer_names:
                raise RuntimeError(
                    f"Internal error: selected partials still mismatch layer_names. File={p.name}"
                )
            if k != k_experts:
                raise RuntimeError(f"k_experts mismatch: {k} vs {k_experts}")

            merged_cnt_topk += cnt_topk
            merged_cnt_top1 += cnt_top1
            merged_cnt_shared += cnt_shared
            merged_tokens_processed += tokens_processed

    assert merged_layer_names is not None

    out_path = domain_dir / "routing_stats_merged.npz"
    np.savez_compressed(
        out_path,
        layer_names=np.array(merged_layer_names, dtype=object),
        cnt_topk=merged_cnt_topk,
        cnt_top1=merged_cnt_top1,
        cnt_shared=merged_cnt_shared,
        tokens_processed=merged_tokens_processed,
        k_experts=np.int64(k_experts),
    )

    # Print sanity checks.
    sum_topk = merged_cnt_topk.sum(axis=1)
    sum_top1 = merged_cnt_top1.sum(axis=1)
    expected_topk = merged_tokens_processed * int(k_experts)

    ok_topk = np.all(sum_topk == expected_topk)
    ok_top1 = np.all(sum_top1 == merged_tokens_processed)

    print(f"[E1.1] Merged {len(selected_items)} partials -> {out_path}")
    print(f"[E1.1] Check topk sum == tokens*k: {ok_topk}")
    print(f"[E1.1] Check top1 sum == tokens: {ok_top1}")

    meta_path = domain_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["merged_npz"] = str(out_path)
        meta["partials_count"] = len(selected_items)
        meta["ignored_partials_count"] = len(partial_files) - len(selected_items)
        meta["sanity_ok_topk"] = bool(ok_topk)
        meta["sanity_ok_top1"] = bool(ok_top1)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

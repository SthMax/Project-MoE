import argparse
import json
import os
import time
from pathlib import Path


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_import(name: str):
    try:
        return __import__(name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Missing dependency '{name}'. Install it in conda env 'moe'.\n"
            f"Example: `pip install {name}`\n"
            f"Original error: {e}"
        ) from e


def load_domain_texts(config: dict, domain: str) -> list[str]:
    domain_cfg = config["domains"].get(domain)
    if domain_cfg is None:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(config['domains'].keys())}")

    datasets = _maybe_import("datasets")
    transformers = _maybe_import("transformers")

    model_id = config["model"]["model_id"]
    trust_remote_code = bool(config["model"].get("trust_remote_code", False))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    data_cfg = config["data"]
    seed = int(data_cfg["seed"])
    n_examples = int(data_cfg["n_examples"])
    max_prompt_tokens = int(data_cfg["max_prompt_tokens"])

    hf = domain_cfg.get("hf_dataset")
    if hf is None:
        raise ValueError(f"Domain {domain} missing hf_dataset config")

    ds = datasets.load_dataset(
        hf["path"],
        hf.get("name"),
        split=hf.get("split", "train"),
    )

    text_field = hf["text_field"]

    # Shuffle deterministically then take n_examples.
    ds = ds.shuffle(seed=seed)

    texts: list[str] = []
    for ex in ds:
        text = ex.get(text_field)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        # Token-level truncation for comparability.
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        if len(ids) == 0:
            continue
        if len(ids) > max_prompt_tokens:
            ids = ids[:max_prompt_tokens]
            text = tokenizer.decode(ids, skip_special_tokens=True)

        texts.append(text)
        if len(texts) >= n_examples:
            break

    if len(texts) < n_examples:
        raise RuntimeError(
            f"Only collected {len(texts)} examples for domain {domain} (requested {n_examples})."
        )

    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()

    config = _load_json(args.config)

    run_id = config.get("run_id") or time.strftime("%Y-%m-%d_%H%M%S")
    domain = args.domain

    output_root = Path(config.get("output_root", "Code/Verification/E1.1/outputs/runs"))
    run_dir = Path("/home/zhhuang/sthmax/git/Project-MoE") / output_root / run_id
    domain_dir = run_dir / domain
    partials_dir = domain_dir / "partials"
    _ensure_dir(partials_dir)

    # Enable vLLM-side routing stats instrumentation.
    os.environ["VLLM_MOE_STATS_DIR"] = str(partials_dir)
    os.environ["VLLM_MOE_STATS_DOMAIN"] = domain
    # Default behavior in vLLM patch: only rank0 writes stats.

    # Load texts.
    prompts = load_domain_texts(config, domain)

    vllm = _maybe_import("vllm")

    model_cfg = config["model"]
    gen_cfg = config["generation"]

    llm = vllm.LLM(
        model=model_cfg["model_id"],
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        max_model_len=int(model_cfg.get("max_model_len", 4096)),
        max_num_seqs=int(model_cfg.get("max_num_seqs", 8)),
        dtype=model_cfg.get("dtype", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    sampling_params = vllm.SamplingParams(
        max_tokens=int(gen_cfg.get("max_tokens", 1)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
    )

    t0 = time.time()
    llm.generate(prompts, sampling_params)
    t1 = time.time()

    # Force flush if the module exists (atexit will also flush).
    try:
        from vllm import moe_stats as _moe_stats

        _moe_stats.flush(reason="collect_routing_stats_end")
    except Exception:
        pass

    meta = {
        "run_id": run_id,
        "domain": domain,
        "model": model_cfg,
        "generation": gen_cfg,
        "data": config["data"],
        "vllm_file": getattr(vllm, "__file__", None),
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "elapsed_sec": t1 - t0,
        "partials_dir": str(partials_dir),
    }

    with open(domain_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[E1.1] Done domain={domain} run_id={run_id}")
    print(f"[E1.1] Partials: {partials_dir}")
    print(f"[E1.1] Meta: {domain_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

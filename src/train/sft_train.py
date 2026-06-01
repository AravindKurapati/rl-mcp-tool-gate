"""Train the gate encoder with supervised contrastive SFT on Modal A10G.

Baseline counterpart to ``src/train/modal_train.py`` (GRPO). Identical model, LoRA
config, data, split, eval, and best-on-val checkpointing — the ONLY difference is the
objective: full-catalog multi-positive InfoNCE (``src/train/sft_loss.py``) instead of
policy gradient. This isolates whether GRPO's reward-shaping bought anything over
copying the ground-truth labels.
"""
from __future__ import annotations
import modal

app = modal.App("rl-mcp-tool-gate-sft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.40",
        "peft>=0.11",
        "accelerate>=0.30",
        "numpy>=1.26",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("data/synthetic", remote_path="/root/data/synthetic")
)

vol = modal.Volume.from_name("rl-mcp-gate-ckpts", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=4 * 3600, volumes={"/ckpts": vol})
def train(
    n_steps: int = 800,
    lr: float = 1e-4,
    batch_size: int = 8,
    lora_r: int = 8,
    lora_alpha: int = 16,
    score_scale: float = 10.0,
    out_dir: str = "/ckpts/sft",
    catalog_file: str = "catalog_v2.json",
    train_file: str = "train_v2.jsonl",
    val_frac: float = 0.1,
    eval_every: int = 25,
):
    import sys, json
    from pathlib import Path
    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import numpy as np

    from src.train.sft_loss import sft_step

    device = "cuda"
    model_name = "BAAI/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name).to(device)

    lcfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(base, lcfg)
    model.train()
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    cat = json.loads(Path(f"/root/data/synthetic/{catalog_file}").read_text(encoding="utf-8"))["tools"]
    tool_names = [t["name"] for t in cat]
    name_to_idx = {n: i for i, n in enumerate(tool_names)}
    catalog_size = len(cat)
    catalog_texts = [t["embed_text"] for t in cat]

    all_data = [json.loads(l) for l in Path(f"/root/data/synthetic/{train_file}").read_text(encoding="utf-8").splitlines()]
    all_data = [t for t in all_data if all(g in name_to_idx for g in t["ground_truth"])]
    # Same val split as GRPO (seed=7) so the early-stopping signal is comparable.
    rng_split = np.random.default_rng(7)
    perm = rng_split.permutation(len(all_data))
    n_val = max(1, int(len(all_data) * val_frac))
    val_data = [all_data[i] for i in perm[:n_val]]
    train_data = [all_data[i] for i in perm[n_val:]]

    def encode_batch(texts, mdl):
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = mdl(**enc)
        cls = out.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, p=2, dim=1)

    def val_recall() -> float:
        # Deterministic top-k recall on the val split — identical metric to GRPO's.
        model.eval()
        with torch.no_grad():
            cat_emb = encode_batch(catalog_texts, model)
            hits = []
            for item in val_data:
                gt = {name_to_idx[g] for g in item["ground_truth"]}
                q = encode_batch([item["query"]], model)
                scores = (q @ cat_emb.T).squeeze(0)
                k = min(catalog_size, max(item["min_k"], 1) + 4)
                topk = set(torch.topk(scores, k).indices.tolist())
                hits.append(len(gt & topk) / len(gt))
        model.train()
        return float(np.mean(hits))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    history = []
    loss_ema = None
    best_val = -1.0
    best_step = -1

    for step in range(n_steps):
        # Encode catalog WITH gradient each step so both sides of the bi-encoder train.
        catalog_emb = encode_batch(catalog_texts, model)

        batch_idx = rng.choice(len(train_data), size=batch_size, replace=False)
        batch = [train_data[i] for i in batch_idx]
        query_texts = [b["query"] for b in batch]
        query_emb_batch = encode_batch(query_texts, model)

        total_loss = torch.tensor(0.0, device=device)
        loss_sum = 0.0
        for bi, item in enumerate(batch):
            gt_idx = {name_to_idx[g] for g in item["ground_truth"]}
            q = query_emb_batch[bi:bi + 1]
            loss, info = sft_step(
                query_emb=q, catalog_emb=catalog_emb,
                ground_truth_idx=gt_idx, score_scale=score_scale,
            )
            total_loss = total_loss + loss
            loss_sum += info["loss"]

        total_loss = total_loss / batch_size
        loss_mean = loss_sum / batch_size

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optim.step()

        loss_ema = loss_mean if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_mean
        rec = {"step": step, "loss": float(total_loss.detach()), "loss_ema": loss_ema}

        if step % eval_every == 0 or step == n_steps - 1:
            vr = val_recall()
            rec["val_recall"] = vr
            if vr > best_val:
                best_val = vr
                best_step = step
                model.save_pretrained(str(out))  # keep the best-on-val checkpoint
                vol.commit()
            print(f"step={step} loss={float(total_loss.detach()):.4f} ema={loss_ema:.3f} "
                  f"val_recall={vr:.3f} (best={best_val:.3f}@{best_step})", flush=True)
        history.append(rec)

    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
    vol.commit()
    print(f"Best val_recall={best_val:.3f} at step {best_step}; checkpoint saved to {out_dir}", flush=True)
    n = len(history)
    return {
        "loss_start": history[min(10, n - 1)]["loss_ema"],
        "loss_end": history[-1]["loss_ema"],
        "best_val_recall": best_val,
        "best_step": best_step,
    }


@app.local_entrypoint()
def smoke(n_steps: int = 150):
    res = train.remote(n_steps=n_steps, out_dir="/ckpts/sft_smoke")
    print("SFT smoke result:", res)


@app.local_entrypoint()
def sft(n_steps: int = 800):
    res = train.remote(n_steps=n_steps, out_dir="/ckpts/sft")
    print("SFT result:", res)

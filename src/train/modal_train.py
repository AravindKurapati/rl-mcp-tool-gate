"""Train the gate encoder with GRPO on Modal A10G."""
from __future__ import annotations
import modal

app = modal.App("rl-mcp-tool-gate-train")

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
    n_steps: int = 1000,
    lr: float = 2e-5,
    batch_size: int = 8,
    n_samples: int = 4,
    kl_coef: float = 0.02,
    lora_r: int = 16,
    lora_alpha: int = 32,
    out_dir: str = "/ckpts/run1",
):
    import sys, json
    from pathlib import Path
    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import numpy as np

    from src.train.grpo import grpo_step

    device = "cuda"
    model_name = "BAAI/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name).to(device)
    ref = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    lcfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(base, lcfg)
    model.train()
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    cat = json.loads(Path("/root/data/synthetic/catalog.json").read_text(encoding="utf-8"))["tools"]
    tool_names = [t["name"] for t in cat]
    name_to_idx = {n: i for i, n in enumerate(tool_names)}
    catalog_size = len(cat)
    catalog_texts = [t["embed_text"] for t in cat]

    train_data = [json.loads(l) for l in Path("/root/data/synthetic/train.jsonl").read_text(encoding="utf-8").splitlines()]
    train_data = [t for t in train_data if all(g in name_to_idx for g in t["ground_truth"])]

    def encode_batch(texts, mdl):
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = mdl(**enc)
        cls = out.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, p=2, dim=1)

    with torch.no_grad():
        ref_catalog_emb = encode_batch(catalog_texts, ref)

    rng = np.random.default_rng(42)
    history = []
    reward_ema = None

    for step in range(n_steps):
        # Encode catalog WITH gradient each step so both sides of the bi-encoder train.
        catalog_emb = encode_batch(catalog_texts, model)

        batch_idx = rng.choice(len(train_data), size=batch_size, replace=False)
        batch = [train_data[i] for i in batch_idx]
        query_texts = [b["query"] for b in batch]
        query_emb_batch = encode_batch(query_texts, model)
        with torch.no_grad():
            ref_query_emb_batch = encode_batch(query_texts, ref)

        total_loss = torch.tensor(0.0, device=device)
        info_acc = {"mean_reward": 0.0, "pg_loss": 0.0, "kl_loss": 0.0}

        for bi, item in enumerate(batch):
            gt_idx = {name_to_idx[g] for g in item["ground_truth"]}
            k_target = max(item["min_k"], 1)
            k = min(catalog_size, k_target + 4)
            q = query_emb_batch[bi:bi + 1]
            ref_q = ref_query_emb_batch[bi:bi + 1]
            ref_scores = (ref_q @ ref_catalog_emb.T).squeeze(0).detach()
            loss, info = grpo_step(
                query_emb=q, catalog_emb=catalog_emb, ground_truth_idx=gt_idx,
                head=None, k=k, n_samples=n_samples, catalog_size=catalog_size,
                k_target=k_target, kl_coef=kl_coef, ref_scores=ref_scores,
            )
            total_loss = total_loss + loss
            for kn in info_acc:
                info_acc[kn] += info.get(kn, 0.0)

        total_loss = total_loss / batch_size
        for kn in info_acc:
            info_acc[kn] /= batch_size

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optim.step()

        reward_ema = info_acc["mean_reward"] if reward_ema is None else 0.95 * reward_ema + 0.05 * info_acc["mean_reward"]
        history.append({"step": step, "loss": float(total_loss.detach()), "reward_ema": reward_ema, **info_acc})
        if step % 20 == 0:
            print(f"step={step} loss={float(total_loss):.4f} mean_r={info_acc['mean_reward']:.3f} ema={reward_ema:.3f} pg={info_acc['pg_loss']:.3f} kl={info_acc['kl_loss']:.3f}", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
    vol.commit()
    print(f"Saved checkpoint to {out_dir}", flush=True)
    # Return reward EMA trajectory (start, mid, end) for the local entrypoint to print
    n = len(history)
    return {
        "ema_start": history[min(10, n - 1)]["reward_ema"],
        "ema_mid": history[n // 2]["reward_ema"],
        "ema_end": history[-1]["reward_ema"],
    }


@app.local_entrypoint()
def smoke(n_steps: int = 150):
    res = train.remote(n_steps=n_steps, lr=1e-4, n_samples=8, out_dir="/ckpts/smoke")
    print("Reward EMA trajectory:", res)


@app.local_entrypoint()
def full():
    res = train.remote(n_steps=1000, lr=1e-4, n_samples=8, out_dir="/ckpts/run1")
    print("Reward EMA trajectory:", res)

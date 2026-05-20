"""Augment 50 seeds to ~400 train queries via Qwen3-1.7B on Modal."""
import json
import modal
from pathlib import Path

app = modal.App("rl-mcp-tool-gate-bootstrap")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.40", "torch>=2.2", "accelerate>=0.30")
)


@app.cls(image=image, gpu="A10G", timeout=1800)
class QwenAugmenter:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    @modal.method()
    def augment(self, seeds: list[dict], n_per_seed: int = 7) -> list[dict]:
        import torch
        out: list[dict] = []
        for seed in seeds:
            prompt = (
                "Rewrite the following user request in 7 different ways. "
                "Keep the EXACT same intent - the same tools must be needed to fulfill it. "
                "Vary wording, formality, sentence length, and level of detail. "
                "Output exactly 7 numbered lines, no other text.\n\n"
                f"Original: {seed['query']}\n\n"
                f"Required tools (do not change these): {', '.join(seed['ground_truth'])}\n\n"
                "Variations:\n"
            )
            msgs = [{"role": "user", "content": prompt}]
            chat_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(chat_text, return_tensors="pt").to("cuda")
            input_ids = enc["input_ids"]
            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids,
                    attention_mask=enc.get("attention_mask"),
                    max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95,
                    pad_token_id=self.tok.eos_token_id,
                )
            text = self.tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # strip leading numbering like "1." or "1)"
                while line and line[0].isdigit():
                    line = line[1:]
                line = line.lstrip(".):- ").strip()
                if len(line) > 10:
                    lines.append(line)
            for variant in lines[:n_per_seed]:
                out.append({
                    "query": variant,
                    "ground_truth": seed["ground_truth"],
                    "category": seed["category"],
                    "min_k": seed["min_k"],
                    "source": "bootstrap",
                    "seed_query": seed["query"],
                })
        return out


@app.local_entrypoint()
def main(seeds_path: str = "data/synthetic/seeds.jsonl", out_path: str = "data/synthetic/train.jsonl"):
    seeds = [json.loads(l) for l in Path(seeds_path).read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(seeds)} seeds, augmenting on A10G...")
    aug = QwenAugmenter()
    augmented = aug.augment.remote(seeds, n_per_seed=7)
    all_train: list[dict] = [{**s, "source": "seed"} for s in seeds] + augmented
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for entry in all_train:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(all_train)} train queries to {out_path}")

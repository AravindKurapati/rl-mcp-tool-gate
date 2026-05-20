"""Qwen2.5-1.5B-Instruct as a tool selector baseline (LLM-as-router)."""
from __future__ import annotations
import json
import re
import modal
from pathlib import Path

app = modal.App("rl-mcp-tool-gate-qwen-baseline")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.40", "torch>=2.2", "accelerate>=0.30")
)


@app.cls(image=image, gpu="A10G", timeout=1800)
class QwenSelector:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
        )

    @modal.method()
    def select(self, queries: list[str], catalog_names: list[str], top_k_hint: int = 10) -> list[list[str]]:
        import torch
        cat_str = ", ".join(catalog_names)
        valid_set = set(catalog_names)
        out: list[list[str]] = []
        for q in queries:
            prompt = (
                "You are a tool router. Given a user query and a list of tool names, "
                "select the relevant subset.\n\n"
                f"Available tools: [{cat_str}]\n\n"
                f"User query: {q}\n\n"
                f"Output ONLY a JSON array of up to {top_k_hint} tool names from the list above, "
                'in order of relevance. Example: ["server.tool1", "server.tool2"]\n'
            )
            msgs = [{"role": "user", "content": prompt}]
            chat_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(chat_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ids = self.model.generate(
                    enc["input_ids"], attention_mask=enc.get("attention_mask"),
                    max_new_tokens=256, do_sample=False, pad_token_id=self.tok.eos_token_id,
                )
            text = self.tok.decode(ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\[(.*?)\]", text, re.DOTALL)
            if not m:
                out.append([])
                continue
            names_raw = re.findall(r'"([^"]+)"', m.group(0))
            out.append([n for n in names_raw if n in valid_set])
        return out


@app.local_entrypoint()
def predict(
    queries_jsonl: str = "data/synthetic/heldout.jsonl",
    catalog_json: str = "data/synthetic/catalog.json",
    out_path: str = "results/qwen_baseline_preds.jsonl",
    top_k: int = 10,
):
    queries_data = [json.loads(l) for l in Path(queries_jsonl).read_text(encoding="utf-8").splitlines()]
    queries = [q["query"] for q in queries_data]
    cat = json.loads(Path(catalog_json).read_text(encoding="utf-8"))["tools"]
    names = [t["name"] for t in cat]
    sel = QwenSelector()
    preds = sel.select.remote(queries, names, top_k_hint=top_k)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for q, pred in zip(queries_data, preds):
            f.write(json.dumps({"query": q["query"], "predicted": pred, "ground_truth": q["ground_truth"]}) + "\n")
    print(f"Wrote {len(preds)} predictions")

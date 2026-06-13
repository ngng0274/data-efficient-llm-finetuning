"""
Alpaca instruction-tuning with Qwen2.5-1.5B + LoRA.

Each group uses the same config for fair comparison:
  A  full      outputs/alpaca_A_full/train.json       (51002)
  B  random    outputs/alpaca_B_random/train.json     (10200)
  C  ifd       outputs/alpaca_C_ifd/train.json        (10200)
  D  confidence outputs/alpaca_D_confidence/train.json (10200)

Shared eval: outputs/alpaca_splits/eval.json (1000)

LoRA adapter saved to: outputs/alpaca_{group}/final_model/
  → for lm-eval: --model_args pretrained=Qwen/Qwen2.5-1.5B,peft=outputs/alpaca_{group}/final_model

Usage:
    python scripts/train_alpaca.py --group B
    python scripts/train_alpaca.py --group C
    python scripts/train_alpaca.py --group A
    python scripts/train_alpaca.py --group D
    python scripts/train_alpaca.py --group B --config configs/alpaca_config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.alpaca_dataset import AlpacaDataset

# ── Group metadata ────────────────────────────────────────────────────────────
GROUP_META = {
    "A": {"train_path": "outputs/alpaca_A_full/train.json",
          "output_dir": "outputs/alpaca_A_full",
          "label":      "A_full   (100%)"},
    "B": {"train_path": "outputs/alpaca_B_random/train.json",
          "output_dir": "outputs/alpaca_B_random",
          "label":      "B_random (random 20%)"},
    "C": {"train_path": "outputs/alpaca_C_ifd/train.json",
          "output_dir": "outputs/alpaca_C_ifd",
          "label":      "C_ifd    (IFD top-20%)"},
    "D": {"train_path": "outputs/alpaca_D_confidence/train.json",
          "output_dir": "outputs/alpaca_D_confidence",
          "label":      "D_conf   (Confidence top-20%)"},
}


# ── Training history callback ─────────────────────────────────────────────────
class HistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_logs: list = []
        self.eval_logs:  list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step  = state.global_step
        epoch = round(state.epoch or 0, 3)
        if "loss" in logs:
            self.train_logs.append({
                "step": step, "epoch": epoch,
                "loss": logs["loss"],
                "lr":   logs.get("learning_rate"),
            })
        if "eval_loss" in logs:
            self.eval_logs.append({
                "step": step, "epoch": epoch,
                "eval_loss": logs["eval_loss"],
            })

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"train": self.train_logs, "eval": self.eval_logs}, f, indent=2)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Training History", fontsize=13)
            if self.train_logs:
                steps  = [d["step"] for d in self.train_logs]
                losses = [d["loss"]  for d in self.train_logs]
                axes[0].plot(steps, losses, color="steelblue", linewidth=1.2)
                axes[0].set_title("Train Loss")
                axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
                axes[0].grid(True, alpha=0.3)
            if self.eval_logs:
                steps  = [d["step"]      for d in self.eval_logs]
                losses = [d["eval_loss"] for d in self.eval_logs]
                axes[1].plot(steps, losses, color="coral", linewidth=1.5,
                             marker="o", markersize=4)
                axes[1].set_title("Eval Loss (val 1000)")
                axes[1].set_xlabel("Step"); axes[1].set_ylabel("Loss")
                axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_model_and_tokenizer(cfg: dict):
    name  = cfg["model"]["name"]
    dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16

    print(f"Loading tokenizer : {name}")
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"

    print(f"Loading model     : {name} [{cfg['model']['dtype']}]")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map={"": 0},
    )
    model.config.use_cache    = False
    model.config.pad_token_id = tok.eos_token_id
    return model, tok


def apply_lora(model, lora_cfg: dict):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        inference_mode=False,
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",  required=True, choices=["A", "B", "C", "D"],
                        help="Which selection group to train")
    parser.add_argument("--config", default="configs/alpaca_config.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    meta = GROUP_META[args.group]

    output_dir   = meta["output_dir"]
    train_path   = meta["train_path"]
    eval_path    = cfg["data"]["eval_path"]
    adapter_path = os.path.join(output_dir, "final_model")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Group      : {meta['label']}")
    print(f"Train data : {train_path}")
    print(f"Eval data  : {eval_path}")
    print(f"Output     : {output_dir}")
    print(f"Adapter    : {adapter_path}")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tok = build_model_and_tokenizer(cfg)
    model      = apply_lora(model, cfg["lora"])
    max_len    = cfg["model"]["max_length"]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_raw = load_json(train_path)
    eval_raw  = load_json(eval_path)

    print(f"\nTokenising train ({len(train_raw)} samples) ...")
    train_ds = AlpacaDataset(train_raw, tok, max_len)
    print(f"Tokenising eval  ({len(eval_raw)} samples) ...")
    eval_ds  = AlpacaDataset(eval_raw, tok, max_len, verbose=False)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    tr = cfg["training"]

    # Adjust eval/save steps: more frequent for small datasets (B/C/D)
    n_train = len(train_ds)
    steps_per_epoch = max(1, n_train // (tr["per_device_train_batch_size"]
                                         * tr["gradient_accumulation_steps"]))
    eval_steps = min(tr["eval_steps"], steps_per_epoch)
    save_steps = eval_steps

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = tr["num_epochs"],
        per_device_train_batch_size = tr["per_device_train_batch_size"],
        gradient_accumulation_steps = tr["gradient_accumulation_steps"],
        learning_rate               = tr["learning_rate"],
        lr_scheduler_type           = tr["lr_scheduler"],
        warmup_ratio                = tr["warmup_ratio"],
        weight_decay                = tr["weight_decay"],
        max_grad_norm               = tr["max_grad_norm"],
        bf16                        = tr["bf16"],
        fp16                        = False,
        logging_steps               = tr["logging_steps"],
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = save_steps,
        save_total_limit            = tr["save_total_limit"],
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_num_workers      = 0,
        dataloader_pin_memory       = False,
        seed                        = args.seed,
        report_to                   = "wandb" if cfg["wandb"]["enabled"] else "none",
        run_name                    = f"alpaca_{args.group}",
        ddp_find_unused_parameters  = False,
    )

    print(f"\nsteps_per_epoch={steps_per_epoch}  eval/save every {eval_steps} steps")

    # ── Trainer ───────────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tok,
        label_pad_token_id  = -100,
        pad_to_multiple_of  = 8,
        return_tensors      = "pt",
    )
    history_cb = HistoryCallback()
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = collator,
        tokenizer     = tok,
        callbacks     = [history_cb],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    # Save only the LoRA adapter weights (for lm-eval --peft flag)
    trainer.model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved → {adapter_path}")

    # ── History ───────────────────────────────────────────────────────────────
    history_cb.save(output_dir)

    # ── Results summary ───────────────────────────────────────────────────────
    best_eval = min((e["eval_loss"] for e in history_cb.eval_logs), default=None)
    results = {
        "group":           args.group,
        "label":           meta["label"],
        "model":           cfg["model"]["name"],
        "n_train":         len(train_ds),
        "n_eval":          len(eval_ds),
        "num_epochs":      tr["num_epochs"],
        "learning_rate":   tr["learning_rate"],
        "max_length":      max_len,
        "best_eval_loss":  best_eval,
        "elapsed_min":     round(elapsed, 1),
        "adapter_path":    adapter_path,
        "lm_eval_cmd": (
            f"lm_eval --model hf \\\n"
            f"  --model_args pretrained={cfg['model']['name']},"
            f"peft={adapter_path},dtype=bfloat16 \\\n"
            f"  --tasks arc_challenge,hellaswag \\\n"
            f"  --device cuda:0 --batch_size 8"
        ),
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Training complete  ({elapsed:.1f} min)")
    print(f"Best eval loss   : {best_eval:.4f}" if best_eval else "")
    print(f"Adapter          : {adapter_path}")
    print(f"Results          : {results_path}")
    print(f"\nlm-eval command:")
    print(f"  {results['lm_eval_cmd']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

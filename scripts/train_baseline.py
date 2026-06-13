"""
Stage 1 baseline: LoRA fine-tuning on noisy Alpaca data.

Usage:
    # Noisy baseline (Experiment A)
    python scripts/train_baseline.py

    # Clean upper bound (Experiment B)
    python scripts/train_baseline.py --no_noise --output_dir outputs/experiment_B

    # Override model
    python scripts/train_baseline.py --model_name Qwen/Qwen2.5-3B-Instruct

    # With ROUGE evaluation after training
    python scripts/train_baseline.py --eval_generation
"""

import argparse
import json
import os
import sys
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

from src.data.dataset import InstructionDataset
from src.data.loader import load_alpaca, split_dataset
from src.data.noise_injector import NoiseInjector
from src.evaluation.metrics import compute_perplexity, compute_rouge


# ── Training History Callback ────────────────────────────────────────────────

class TrainingHistoryCallback(TrainerCallback):
    """학습 중 train/eval loss를 수집하고 완료 시 JSON + PNG로 저장."""

    def __init__(self):
        self.train_logs = []   # {"step", "epoch", "loss", "learning_rate", "grad_norm"}
        self.eval_logs  = []   # {"step", "epoch", "eval_loss"}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step  = state.global_step
        epoch = round(state.epoch or 0, 3)

        if "loss" in logs:
            self.train_logs.append({
                "step"         : step,
                "epoch"        : epoch,
                "loss"         : logs["loss"],
                "learning_rate": logs.get("learning_rate"),
                "grad_norm"    : logs.get("grad_norm"),
            })

        if "eval_loss" in logs:
            self.eval_logs.append({
                "step"     : step,
                "epoch"    : epoch,
                "eval_loss": logs["eval_loss"],
            })

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._save_json(output_dir)
        self._save_plot(output_dir)

    def _save_json(self, output_dir: str):
        path = os.path.join(output_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"train": self.train_logs, "eval": self.eval_logs}, f, indent=2)
        print(f"  Log saved  → {path}")

    def _save_plot(self, output_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")   # 헤드리스 환경 대응 (GUI 불필요)
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib 없음 — 그래프 저장 스킵 (pip install matplotlib)")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History", fontsize=13)

        # ── Train loss ───────────────────────────────────────────────────────
        if self.train_logs:
            steps  = [d["step"] for d in self.train_logs]
            losses = [d["loss"]  for d in self.train_logs]
            axes[0].plot(steps, losses, color="steelblue", linewidth=1.2, label="train loss")
            axes[0].set_title("Train Loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # ── Eval loss ────────────────────────────────────────────────────────
        if self.eval_logs:
            steps  = [d["step"]      for d in self.eval_logs]
            losses = [d["eval_loss"] for d in self.eval_logs]
            axes[1].plot(steps, losses, color="coral", linewidth=1.5,
                         marker="o", markersize=5, label="eval loss")
            axes[1].set_title("Eval Loss")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "loss_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]
    dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # must be right for training

    print(f"Loading model: {model_name} [{cfg['model']['dtype']}]")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": 0},   # pin to GPU 0 (single-GPU training)
    )
    model.config.use_cache = False            # required with gradient checkpointing
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


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
    model.enable_input_require_grads()        # needed with gradient_checkpointing
    model.print_trainable_parameters()
    return model


def prepare_data(cfg: dict, no_noise: bool = False):
    data_cfg = cfg["data"]
    print(f"Loading Alpaca ({data_cfg['num_samples']} samples) ...")
    samples = load_alpaca(num_samples=data_cfg["num_samples"], seed=data_cfg["seed"])

    train_raw, val_raw, test_raw = split_dataset(
        samples,
        val_ratio=data_cfg.get("val_ratio", 0.1),
        test_ratio=data_cfg.get("test_ratio", 0.1),
        n_val=data_cfg.get("n_val"),
        n_test=data_cfg.get("n_test"),
        seed=data_cfg["seed"],
    )

    if no_noise:
        print("Noise injection SKIPPED (clean upper bound mode).")
        for s in train_raw:
            s["is_clean"] = True
            s["noise_type"] = "none"
        train_samples = train_raw
    else:
        print(f"Injecting noise (ratio={data_cfg['noise_ratio']}) ...")
        injector = NoiseInjector(
            noise_ratio=data_cfg["noise_ratio"],
            noise_types=data_cfg["noise_types"],
            seed=data_cfg["seed"],
        )
        train_samples = injector.inject(train_raw)
        stats = injector.stats(train_samples)
        print(
            f"  clean={stats['clean']}  noisy={stats['noisy']}  "
            f"({stats['noise_ratio']*100:.0f}% noise)"
        )
        for t, cnt in stats["by_type"].items():
            if t != "none":
                print(f"    {t}: {cnt}")

    return train_samples, val_raw, test_raw


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_config.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_noise", action="store_true",
                        help="Skip noise injection (clean upper bound, Experiment B)")
    parser.add_argument("--eval_generation", action="store_true",
                        help="Run ROUGE evaluation after training (slow)")
    parser.add_argument("--rouge_samples", type=int, default=200,
                        help="Number of test samples to use for ROUGE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    output_dir = cfg["training"]["output_dir"]

    # --no_noise 시 경로/이름의 _noisy 부분을 _clean으로 교체 (어디에 있든)
    if args.no_noise:
        if "_noisy" in output_dir:
            output_dir = output_dir.replace("_noisy", "_clean", 1)
        elif not output_dir.endswith("_clean"):
            output_dir += "_clean"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg["lora"])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_samples, val_raw, test_raw = prepare_data(cfg, no_noise=args.no_noise)

    max_len = cfg["model"]["max_length"]
    print("Tokenising train ...")
    train_dataset = InstructionDataset(train_samples, tokenizer, max_len)
    print("Tokenising val/test ...")
    val_dataset = InstructionDataset(val_raw, tokenizer, max_len, verbose=False)
    test_dataset = InstructionDataset(test_raw, tokenizer, max_len, verbose=False)

    print(f"  train: {len(train_dataset)} ({train_dataset.n_clean} clean, {train_dataset.n_noisy} noisy)")
    print(f"  val  : {len(val_dataset)}")
    print(f"  test : {len(test_dataset)}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    tr_cfg = cfg["training"]
    experiment_name = tr_cfg["experiment_name"]
    if args.no_noise:
        if "_noisy" in experiment_name:
            experiment_name = experiment_name.replace("_noisy", "_clean", 1)
        elif not experiment_name.endswith("_clean"):
            experiment_name += "_clean"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tr_cfg["num_epochs"],
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        learning_rate=tr_cfg["learning_rate"],
        lr_scheduler_type=tr_cfg["lr_scheduler"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        weight_decay=tr_cfg["weight_decay"],
        max_grad_norm=tr_cfg["max_grad_norm"],
        bf16=tr_cfg["bf16"],
        fp16=False,
        logging_steps=tr_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tr_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=tr_cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,   # 0 required on Windows
        dataloader_pin_memory=False,
        report_to="wandb" if cfg["wandb"]["enabled"] else "none",
        run_name=experiment_name,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    history_cb = TrainingHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[history_cb],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Experiment : {experiment_name}")
    print(f"Model      : {cfg['model']['name']}")
    print(f"Noise ratio: {'0% (clean)' if args.no_noise else str(int(cfg['data']['noise_ratio']*100))+'%'}")
    print(f"Output dir : {output_dir}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))

    # ── 학습 기록 저장 ────────────────────────────────────────────────────────
    print("\nSaving training history ...")
    history_cb.save(output_dir)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\nEvaluating ...")
    ppl = compute_perplexity(model, test_dataset, tokenizer, batch_size=4)
    print(f"Test Perplexity: {ppl:.4f}")

    results = {
        "experiment": experiment_name,
        "model": cfg["model"]["name"],
        "noise_ratio": 0.0 if args.no_noise else cfg["data"]["noise_ratio"],
        "test_perplexity": ppl,
    }

    if args.eval_generation:
        n = min(args.rouge_samples, len(test_raw))
        print(f"Computing ROUGE on {n} test samples ...")
        rouge = compute_rouge(model, tokenizer, test_raw[:n], batch_size=4)
        print(f"ROUGE: {rouge}")
        results.update(rouge)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()

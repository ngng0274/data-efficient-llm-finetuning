"""
GSM8K Stage 1 fine-tuning: noisy vs. clean baseline.

Usage:
    # Noisy 50% (Experiment A)
    python scripts/train_gsm8k.py

    # Clean 100% (Experiment B)
    python scripts/train_gsm8k.py --no_noise

    # Smoke test (8 epoch)
    python scripts/train_gsm8k.py --config configs/gsm8k_smoke_test.yaml
    python scripts/train_gsm8k.py --config configs/gsm8k_smoke_test.yaml --no_noise
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

from src.data.gsm8k_dataset import GSM8KDataset
from src.data.gsm8k_loader import load_gsm8k
from src.data.gsm8k_noise import GSM8KAnswerCorruptor
from src.evaluation.gsm8k_eval import compute_gsm8k_accuracy
from src.evaluation.metrics import compute_perplexity


# ── Training History Callback (loss curves) ──────────────────────────────────

class TrainingHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_logs = []
        self.eval_logs  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step  = state.global_step
        epoch = round(state.epoch or 0, 3)
        if "loss" in logs:
            self.train_logs.append({
                "step": step, "epoch": epoch,
                "loss": logs["loss"],
                "learning_rate": logs.get("learning_rate"),
                "grad_norm":     logs.get("grad_norm"),
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
        print(f"  Log saved  → {path}")
        self._save_plot(output_dir)

    def _save_plot(self, output_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History", fontsize=13)
        if self.train_logs:
            steps  = [d["step"] for d in self.train_logs]
            losses = [d["loss"]  for d in self.train_logs]
            axes[0].plot(steps, losses, color="steelblue", linewidth=1.2)
            axes[0].set_title("Train Loss"); axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)
        if self.eval_logs:
            steps  = [d["step"]      for d in self.eval_logs]
            losses = [d["eval_loss"] for d in self.eval_logs]
            axes[1].plot(steps, losses, color="coral", linewidth=1.5,
                         marker="o", markersize=5)
            axes[1].set_title("Eval Loss (val)"); axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Loss"); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "loss_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")


# ── Epoch Accuracy Callback (per-epoch test accuracy) ────────────────────────

class EpochAccuracyCallback(TrainerCallback):
    """Runs test-set exact-match accuracy after every epoch.

    Saves results to epoch_accuracy.json (appended each epoch so the file
    survives an early-stop or crash) and generates epoch_accuracy.png on
    completion.
    """

    def __init__(
        self,
        test_samples: list,
        tokenizer,
        eval_cfg: dict,
        output_dir: str,
        experiment_name: str,
        noise_ratio: float,
    ):
        self.test_samples    = test_samples
        self.tokenizer       = tokenizer
        self.eval_cfg        = eval_cfg
        self.output_dir      = output_dir
        self.experiment_name = experiment_name
        self.noise_ratio     = noise_ratio
        self.history: list   = []   # [{epoch, accuracy, n_correct, n_total, n_no_answer}]

    # called by Trainer at the end of every epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        epoch = int(round(state.epoch))

        print(f"\n[Epoch {epoch}/{int(state.num_train_epochs)}] "
              f"Evaluating test accuracy ...")

        summary = compute_gsm8k_accuracy(
            model,
            self.tokenizer,
            self.test_samples,
            max_new_tokens=self.eval_cfg.get("max_new_tokens", 512),
            batch_size=self.eval_cfg.get("batch_size", 4),
            output_dir=None,   # no per-epoch prediction files
            split_name=f"epoch{epoch}",
        )

        record = {
            "epoch":       epoch,
            "accuracy":    summary["accuracy"],
            "n_correct":   summary["n_correct"],
            "n_total":     summary["n_total"],
            "n_no_answer": summary["n_no_answer"],
        }
        self.history.append(record)
        print(f"  → Accuracy: {summary['accuracy']:.2f}%  "
              f"({summary['n_correct']}/{summary['n_total']})")

        self._save_json()   # flush after each epoch
        return control

    def _save_json(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "epoch_accuracy.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment":   self.experiment_name,
                    "noise_ratio":  self.noise_ratio,
                    "eval_samples": len(self.test_samples),  # subset size per epoch
                    "epochs":       self.history,
                },
                f, indent=2,
            )

    def save_plot(self):
        path = os.path.join(self.output_dir, "epoch_accuracy.json")
        print(f"  Epoch accuracy → {path}")
        if not self.history:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not found — skipping accuracy plot")
            return

        epochs = [r["epoch"]    for r in self.history]
        accs   = [r["accuracy"] for r in self.history]
        color  = "coral" if self.noise_ratio > 0 else "steelblue"
        label  = f"Noisy ({int(self.noise_ratio*100)}%)" if self.noise_ratio > 0 else "Clean (100%)"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, accs, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.set_title(f"Test Accuracy per Epoch\n{self.experiment_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(epochs)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "epoch_accuracy.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Accuracy plot → {plot_path}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    name  = cfg["model"]["name"]
    dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16
    print(f"Loading tokenizer: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Loading model: {name} [{cfg['model']['dtype']}]")
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map={"": 0},
    )
    model.config.use_cache    = False
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
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def prepare_data(cfg: dict, no_noise: bool = False):
    d = cfg["data"]
    print(f"Loading GSM8K (train={d['num_train']}, val={d['num_val']}, "
          f"test={d['num_test']}) ...")
    train_raw, val_raw, test_raw = load_gsm8k(
        num_train=d["num_train"],
        num_val=d["num_val"],
        num_test=d["num_test"],
        seed=d["seed"],
    )
    if no_noise:
        print("Noise injection SKIPPED (clean upper bound).")
        for s in train_raw:
            s["is_noisy"]  = False
            s["is_clean"]  = True
            s["noise_type"] = "none"
        return train_raw, val_raw, test_raw

    print(f"Injecting noise (ratio={d['noise_ratio']}) ...")
    corruptor = GSM8KAnswerCorruptor(noise_ratio=d["noise_ratio"], seed=d["seed"])
    train_samples = corruptor.inject(train_raw)
    stats = corruptor.stats(train_samples)
    print(f"  clean={stats['clean']}  noisy={stats['noisy']}  "
          f"({stats['noise_ratio']*100:.0f}% noise)")
    return train_samples, val_raw, test_raw


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/gsm8k_config.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_noise",   action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    output_dir = cfg["training"]["output_dir"]
    if args.no_noise:
        output_dir = (output_dir.replace("_noisy", "_clean", 1)
                      if "_noisy" in output_dir
                      else output_dir + "_clean")

    experiment_name = cfg["training"]["experiment_name"]
    if args.no_noise:
        experiment_name = (experiment_name.replace("_noisy", "_clean", 1)
                           if "_noisy" in experiment_name
                           else experiment_name + "_clean")

    noise_ratio = 0.0 if args.no_noise else cfg["data"]["noise_ratio"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg["lora"])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_samples, val_raw, test_raw = prepare_data(cfg, no_noise=args.no_noise)

    max_len = cfg["model"]["max_length"]
    print("Tokenising train ...")
    train_dataset = GSM8KDataset(train_samples, tokenizer, max_len)
    print("Tokenising val/test ...")
    val_dataset  = GSM8KDataset(val_raw,  tokenizer, max_len, verbose=False)
    test_dataset = GSM8KDataset(test_raw, tokenizer, max_len, verbose=False)
    print(f"  train: {len(train_dataset)} "
          f"({train_dataset.n_clean} clean, {train_dataset.n_noisy} noisy)")
    print(f"  val  : {len(val_dataset)}")
    print(f"  test : {len(test_dataset)}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    tr = cfg["training"]
    eval_cfg = cfg.get("eval", {})

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tr["num_epochs"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler"],
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        bf16=tr["bf16"],
        fp16=False,
        logging_steps=tr["logging_steps"],
        eval_strategy="steps",
        eval_steps=tr["eval_steps"],
        save_strategy="steps",
        save_steps=tr["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="wandb" if cfg["wandb"]["enabled"] else "none",
        run_name=experiment_name,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100,
        pad_to_multiple_of=8, return_tensors="pt",
    )

    # Per-epoch eval uses a small subset for speed; final eval uses full test set.
    epoch_subset = eval_cfg.get("epoch_subset", len(test_raw))
    epoch_test_samples = test_raw[:epoch_subset]

    history_cb  = TrainingHistoryCallback()
    accuracy_cb = EpochAccuracyCallback(
        test_samples=epoch_test_samples,
        tokenizer=tokenizer,
        eval_cfg=eval_cfg,
        output_dir=output_dir,
        experiment_name=experiment_name,
        noise_ratio=noise_ratio,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[history_cb, accuracy_cb],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Experiment : {experiment_name}")
    print(f"Model      : {cfg['model']['name']}")
    print(f"Epochs     : {tr['num_epochs']}")
    print(f"Noise      : {'0% (clean)' if args.no_noise else str(int(noise_ratio*100))+'%'}")
    print(f"Output     : {output_dir}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))

    # ── Save training history & accuracy curves ───────────────────────────────
    print("\nSaving training history ...")
    history_cb.save(output_dir)

    print("\nSaving epoch accuracy ...")
    accuracy_cb.save_plot()

    # ── Final evaluation (best model, loaded by load_best_model_at_end) ───────
    print("\n[1/2] Perplexity (test, response tokens) ...")
    ppl = compute_perplexity(model, test_dataset, tokenizer, batch_size=4)
    print(f"  Test Perplexity: {ppl:.4f}")

    print("\n[2/2] Accuracy (exact match, best checkpoint) ...")
    acc_summary = compute_gsm8k_accuracy(
        model, tokenizer, test_raw,
        max_new_tokens=eval_cfg.get("max_new_tokens", 512),
        batch_size=eval_cfg.get("batch_size", 4),
        output_dir=output_dir,
        split_name="test",
    )
    print(f"  Accuracy : {acc_summary['accuracy']:.2f}%  "
          f"({acc_summary['n_correct']}/{acc_summary['n_total']})")
    if acc_summary["n_no_answer"] > 0:
        print(f"  No-answer: {acc_summary['n_no_answer']} samples")

    # ── Combined results.json ─────────────────────────────────────────────────
    results = {
        "experiment":      experiment_name,
        "model":           cfg["model"]["name"],
        "noise_ratio":     noise_ratio,
        "num_epochs":      tr["num_epochs"],
        "test_perplexity": ppl,
        **acc_summary,
        "epoch_accuracy":  accuracy_cb.history,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    return results


if __name__ == "__main__":
    main()

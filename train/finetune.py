"""
Fine-tune Llama on chess by next-move prediction over move sequences
(completion-style on the `text` field = clean SAN movetext). Base model (not
Instruct) — we want a pure PGN continuer.

Runs on the GPU box. Two modes:
  LoRA (spike):  python train/finetune.py --merge
  Full FT:       python train/finetune.py --full --merge   (needs a big GPU)

Outputs:
  adapters/chess-ft/        LoRA adapter (LoRA mode)
  merged-model/chess-ft/    merged fp16 weights for vLLM (with --merge, or full)
"""
import argparse
import glob
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DEFAULT = "unsloth/Meta-Llama-3.1-8B"   # BASE (completion), not Instruct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--train-file", default="data/train.jsonl",
                    help="training jsonl (data/train.jsonl=SAN, data/train_uci.jsonl=UCI)")
    ap.add_argument("--out-name", default="chess-ft",
                    help="adapter/merged dir name under adapters/ and merged-model/")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=1024)  # movetext is short
    ap.add_argument("--full", action="store_true", help="full fine-tune (big GPU)")
    ap.add_argument("--packing", action="store_true",
                    help="pack multiple short games per sequence (~7x throughput on short movetext)")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="checkpoint every N steps (keep only latest); 0 = no checkpoints. "
                         "Enables resume-on-restart if the run dies with disk intact.")
    ap.add_argument("--out-dir", default="outputs",
                    help="trainer output_dir for checkpoints (resume reads from here)")
    ap.add_argument("--merge", action="store_true", help="export merged fp16 for vLLM")
    ap.add_argument("--hf-repo", default=None,
                    help="if set, push the merged fp16 model to this HF repo the instant "
                         "the merge finishes (uses HF_TOKEN env)")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=not args.full,          # 4bit QLoRA for spike; bf16 for full FT
        full_finetuning=args.full,
    )
    if args.full:
        # Full-FT of 8B won't fit 80GB without activation checkpointing; the LoRA
        # branch gets this via get_peft_model, so the full path must enable it too.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if not args.full:
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=42,
        )

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else str(ROOT / args.out_dir)

    train_ds = load_dataset("json", data_files=str(ROOT / args.train_file),
                            split="train")
    print(f"training on {len(train_ds)} games from {args.train_file} "
          f"({'full FT' if args.full else f'LoRA r{args.lora_rank}'}) -> out_dir={out_dir}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        dataset_text_field="text", max_seq_length=args.max_seq_len,
        args=SFTConfig(
            packing=args.packing,            # concat short games -> full windows, far fewer steps
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            warmup_ratio=0.03, logging_steps=20, optim="adamw_8bit",
            lr_scheduler_type="cosine", seed=42, bf16=True,
            # Checkpointing: a full-FT 8B checkpoint is ~32GB (model + optimizer),
            # so keep ONLY the latest (save_total_limit=1) to bound disk — several
            # checkpoints once filled the disk and crashed the save. With --save-steps
            # we get resume-on-restart; with 0 we keep the old no-checkpoint behavior.
            save_strategy=("steps" if args.save_steps > 0 else "no"),
            save_steps=(args.save_steps if args.save_steps > 0 else 500),
            save_total_limit=1,
            output_dir=out_dir, report_to="none",
        ),
    )
    # Resume from the latest checkpoint if one survived on disk (e.g. pod stopped
    # and was relaunched). HF Trainer picks the newest checkpoint-* in out_dir.
    ckpts = glob.glob(os.path.join(out_dir, "checkpoint-*"))
    resume = max(ckpts, key=lambda c: int(re.search(r"checkpoint-(\d+)", c).group(1))) if ckpts else None
    print(f"RESUMING from {resume}" if resume else "training from scratch (no checkpoint found)")
    trainer.train(resume_from_checkpoint=resume)

    if not args.full:
        adapter = ROOT / "adapters" / args.out_name
        model.save_pretrained(str(adapter)); tokenizer.save_pretrained(str(adapter))
        print(f"saved LoRA adapter -> {adapter}")
    if args.merge or args.full:
        merged = ROOT / "merged-model" / args.out_name
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
        print(f"saved merged fp16 model -> {merged}")
        # Push to HF the INSTANT the merge finishes, before eval/teardown — so the
        # weights are preserved off-pod even if anything downstream dies.
        if args.hf_repo:
            print(f"PUSHING merged fp16 model to HF repo {args.hf_repo} ...")
            model.push_to_hub_merged(args.hf_repo, tokenizer, save_method="merged_16bit",
                                     token=os.environ.get("HF_TOKEN"))
            print(f"PUSHED_TO_HF {args.hf_repo}")


if __name__ == "__main__":
    main()

# LLaVA-Med Pruning Evaluation Harness

Evaluation infrastructure for **question-aware visual token pruning** research
on medical vision-language models, using **LLaVA-Med v1.5 (Mistral-7B)** as the
baseline. Decoupled from the LLaVA-Med codebase; imports `llava` as a library.

This repository is the *measurement infrastructure* for the project — it runs a
model over a medical VQA benchmark, optionally with a pruning method attached,
and reports accuracy, latency, and memory. The pruning methods themselves are
the research contribution and are added on top of this harness.

## Status

The harness is functional end-to-end for VQA-RAD. SLAKE and PathVQA loaders are
stubbed (interfaces defined, implementation pending).

- [x] Dataset / method base interfaces
- [x] Metrics module (closed/open accuracy, latency, peak memory)
- [x] Model loader wrapper (LLaVA-Med v1.5)
- [x] Runner (evaluation orchestrator)
- [x] CLI entry point
- [x] Baseline (no-op) method
- [x] VQA-RAD loader — with real answer_type labels joined from the
      original VQA-RAD distribution
- [ ] SLAKE loader (stub)
- [ ] PathVQA loader (stub)
- [ ] Pruning methods (the research contribution — not yet started)

**Verification:** harness inference output has been confirmed to match the
reference llava/eval/model_vqa.py verbatim on sample questions.

**E00 baseline** (unmodified LLaVA-Med v1.5, VQA-RAD test set, 451 samples):
closed-ended accuracy 0.537, open-ended recall 0.340.

## Layout

    llava-med-pruning/
    ├── eval/
    │   ├── runner.py            # Main entry: run_eval(...) -> dict
    │   ├── metrics.py           # Closed/open accuracy, latency, peak memory
    │   ├── model_loader.py      # Wraps llava.model.builder.load_pretrained_model
    │   ├── datasets/
    │   │   ├── base.py          # MedVQADataset interface + VQASample dataclass
    │   │   ├── vqa_rad.py       # VQA-RAD loader (implemented)
    │   │   ├── slake.py         # SLAKE loader (stub)
    │   │   └── path_vqa.py      # PathVQA loader (stub)
    │   └── methods/
    │       ├── base.py          # PruningMethod interface
    │       └── baseline.py      # No-op method (unmodified model)
    ├── scripts/
    │   └── run_eval.py          # CLI entry point (argparse)
    └── results/                 # Output files (gitignored — regenerable)

## Output format

Each evaluation run writes two files to `results/`:

1. `<run_id>_metrics.json` — aggregate metrics plus the fully resolved config
   (every CLI argument is saved, so a run is reproducible from this file alone).
2. `<run_id>_predictions.jsonl` — one JSON object per evaluated question, for
   error analysis and qualitative inspection.

`<run_id>` is `{method}_{dataset}_kr{keep_ratio}_{timestamp}`.

## Usage

    python scripts/run_eval.py \
        --model-path /data/dan/weights/llava-med-v1.5-mistral-7b \
        --dataset vqa_rad \
        --dataset-root /data/dan/dataset/vqa_rad \
        --method baseline \
        --output-dir results/

Add `--max-samples N` to evaluate only the first N samples (useful for quick
smoke tests during development).

## Notes on the VQA-RAD loader

The harness loads VQA-RAD from the HuggingFace Parquet mirror
(`flaviagiammarino/vqa-rad`), which does not carry the original dataset's
`answer_type` field. Real closed/open labels are restored by joining against
the original VQA-RAD distribution (`VQA_RAD Dataset Public.json`) on a
normalized `(question, answer)` key. As of the current dataset copies, 450/451
test samples join successfully; 1 falls back to a heuristic label (documented
in `vqa_rad.py`).

Parquet-embedded images are materialized to disk once, on first load, under
`<dataset-root>/extracted_images/`.

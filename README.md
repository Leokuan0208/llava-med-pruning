# LLaVA-Med Pruning Evaluation Harness

Evaluation infrastructure for question-aware visual token pruning research on
medical VLMs. Decoupled from the LLaVA-Med codebase; imports `llava` as a
library.

## Layout

```
llava-med-pruning/
├── eval/
│   ├── runner.py            # Main entry: run_eval(...) -> dict
│   ├── metrics.py           # Accuracy, latency, peak memory measurements
│   ├── model_loader.py      # Wraps llava.model.builder.load_pretrained_model
│   ├── datasets/
│   │   ├── base.py          # MedVQADataset interface
│   │   ├── vqa_rad.py       # VQA-RAD loader
│   │   ├── slake.py         # SLAKE loader
│   │   └── path_vqa.py      # PathVQA loader
│   └── methods/
│       ├── base.py          # PruningMethod interface
│       └── baseline.py      # No-op (unmodified model)
├── scripts/
│   ├── run_eval.py          # CLI entry point (argparse)
│   └── run_sweep.sh         # Shell loop for parameter sweeps
├── configs/
│   └── README.md            # Notes on argument conventions
└── results/
    └── (output JSONs and JSONL prediction files)
```

## Output format

Each evaluation run produces two files in `results/`:

1. **`<run_id>_metrics.json`** — aggregate metrics + the resolved config
   (everything from `vars(args)` is saved here so runs are reproducible).
2. **`<run_id>_predictions.jsonl`** — one JSON object per evaluated question,
   for error analysis and qualitative figures.

`<run_id>` is `{method}_{dataset}_{keep_ratio}_{timestamp}`.

## Usage

```bash
python scripts/run_eval.py \
    --model-path /data/dan/weights/llava-med-v1.5-mistral-7b \
    --dataset vqa_rad \
    --method baseline \
    --output-dir results/
```

## Status

- [x] Directory structure
- [ ] Dataset base interface
- [ ] VQA-RAD loader
- [ ] SLAKE loader
- [ ] PathVQA loader
- [ ] Method base interface
- [ ] Baseline (no-op) method
- [ ] Metrics module
- [ ] Model loader wrapper
- [ ] Runner
- [ ] CLI entry point

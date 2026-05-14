"""Main evaluation orchestrator.

run_eval(loaded_model, method, dataset, ...) is the one function that
ties the harness together. It iterates over the dataset, runs the model
(with the method attached), collects per-question predictions, computes
aggregate metrics, and returns a result dict.

The per-sample generation logic mirrors the verified-working inference
path from the patched llava/serve/cli.py, with three deliberate changes
for batch evaluation:

  1. Single-turn, not multi-turn: a fresh conversation is built per
     sample. Carrying conversation state across samples would leak one
     question's answer into the next question's context.
  2. Fresh image per sample: process_images moves inside the loop,
     because each dataset row has its own image.
  3. Silent and deterministic: no TextStreamer, and greedy decoding
     (temperature 0.0) by default for reproducible metrics.
"""

import time
from typing import Optional

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from PIL import Image

from .model_loader import LoadedModel
from .methods.base import PruningMethod
from .datasets.base import MedVQADataset
from .metrics import (
    closed_ended_accuracy,
    open_ended_recall,
    LatencyTracker,
    peak_gpu_memory_gb,
    reset_peak_memory,
)


def _build_prompt(question: str, conv_mode: str, mm_use_im_start_end: bool) -> str:
    """Construct the full prompt string for one question.

    Mirrors cli.py's first-message branch: the image placeholder token is
    prepended to the question text, then wrapped in the conversation
    template. A fresh conv object is created every call -- this is the
    single-turn behavior that keeps samples independent.

    Args:
        question: The raw question text from the VQASample.
        conv_mode: Conversation template name (e.g. "mistral_instruct").
        mm_use_im_start_end: model.config.mm_use_im_start_end -- whether
            the image token needs explicit start/end markers around it.

    Returns:
        The assembled prompt string, ready for tokenizer_image_token.
    """
    # Fresh conversation per call. .copy() gives an independent object so
    # nothing from a previous sample lingers.
    conv = conv_templates[conv_mode].copy()

    # Prepend the image placeholder to the question, exactly as cli.py does
    # for its first message.
    if mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv.append_message(conv.roles[0], inp)   # user turn
    conv.append_message(conv.roles[1], None)  # assistant turn, empty -> model fills it
    return conv.get_prompt()


def _generate_one(
    loaded_model: LoadedModel,
    question: str,
    image_path: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run the model on a single (question, image) pair, return the answer text.

    This is the inner step of the evaluation loop. The sequence of calls
    is lifted from the patched cli.py and verified to produce coherent
    biomedical responses.

    Args:
        loaded_model: The LoadedModel bundle.
        question: Raw question text.
        image_path: Absolute path to the image file on disk.
        max_new_tokens: Generation length cap.
        temperature: 0.0 -> greedy/deterministic; >0 -> sampling.

    Returns:
        The decoded answer string, stripped of the stop token.
    """
    model = loaded_model.model
    tokenizer = loaded_model.tokenizer
    image_processor = loaded_model.image_processor

    # --- Prompt construction -------------------------------------------------
    prompt = _build_prompt(
        question,
        loaded_model.conv_mode,
        model.config.mm_use_im_start_end,
    )

    # --- Image processing (fresh per sample) ---------------------------------
    # Open the file, then run it through the vision-tower preprocessor.
    # The list-vs-tensor branch and the float16 cast are exactly as cli.py
    # handles them.
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # --- Tokenize the prompt -------------------------------------------------
    # tokenizer_image_token handles the special IMAGE_TOKEN_INDEX placeholder
    # that a plain tokenizer would not understand.
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    # --- Stopping criteria ---------------------------------------------------
    # This is the patched line: it handles SeparatorStyle.LLAMA_2 (used by
    # mistral_instruct), which the original cli.py did not.
    stop_str = conv_templates[loaded_model.conv_mode].copy().sep2 \
        if conv_templates[loaded_model.conv_mode].sep_style in (SeparatorStyle.TWO, SeparatorStyle.LLAMA_2) \
        else conv_templates[loaded_model.conv_mode].sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # --- Generate ------------------------------------------------------------
    # No TextStreamer (silent). do_sample follows temperature, same pattern
    # as cli.py. torch.inference_mode() disables gradient tracking -- we're
    # only doing forward passes, so this saves memory and time.
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # --- Decode --------------------------------------------------------------
    # IMPORTANT: this model's generate() returns ONLY the generated tokens,
    # NOT [prompt + generated]. This is because the <image> placeholder (a
    # single IMAGE_TOKEN_INDEX token in input_ids) is expanded into ~576
    # visual embeddings inside the forward pass, so there is no clean token-ID
    # prefix to return. Verified empirically: output_ids contains exactly the
    # answer, bracketed by BOS (1) and EOS (2).
    #
    # The base-LLaVA convention `output_ids[0, input_ids.shape[1]:]` is WRONG
    # here -- it slices past the end of a tensor that has no prompt prefix,
    # producing an empty string. cli.py carries this latent bug but masks it
    # with a TextStreamer; the harness has no streamer, so it must decode
    # output_ids directly.
    #
    # skip_special_tokens=True drops BOS/EOS (and the stop token, which is a
    # special token) cleanly, so no manual stop_str stripping is needed.
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return outputs


def run_eval(
    loaded_model: LoadedModel,
    method: PruningMethod,
    dataset: MedVQADataset,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 42,
) -> dict:
    """Run one (model, method, dataset) evaluation.

    Args:
        loaded_model: From eval.model_loader.load_llava_med.
        method: From eval.methods.*. Determines pruning behavior.
        dataset: From eval.datasets.*. Provides VQASample iterator.
        max_new_tokens: Generation length cap. 128 is plenty for medical
            VQA (most answers are <50 tokens); larger wastes compute.
        temperature: 0.0 = greedy. Use greedy for reproducible eval;
            sampling is only useful when you want answer diversity.
        seed: For any sampling-based behavior.

    Returns:
        A dict with three top-level keys:
            "metrics": aggregate numbers (accuracy, latency, memory).
            "predictions": list of per-question result dicts.
            "config": resolved hyperparameters for reproducibility.
    """
    # Seed everything that could introduce nondeterminism. With temperature
    # 0.0 generation is already greedy, but seeding keeps any incidental
    # randomness (dropout-in-eval edge cases, etc.) reproducible.
    torch.manual_seed(seed)

    # --- Attach the pruning method ------------------------------------------
    # For BaselineMethod this is a no-op; for real methods it installs hooks.
    method.attach(loaded_model.model)

    # --- Set up measurement -------------------------------------------------
    reset_peak_memory()
    tracker = LatencyTracker(warmup=5)

    predictions = []

    # --- Main evaluation loop -----------------------------------------------
    total = len(dataset)
    try:
        for i, sample in enumerate(dataset, start=1):
            tracker.start()
            pred_text = _generate_one(
                loaded_model,
                sample.question,
                sample.image_path,
                max_new_tokens,
                temperature,
            )
            tracker.stop()

            predictions.append({
                "question_id": sample.question_id,
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": pred_text,
                "answer_type": sample.answer_type,
                "dataset": sample.dataset,
            })

            # Progress line. \r returns the cursor to the line start so each
            # update overwrites the previous one instead of scrolling -- the
            # terminal shows a single advancing counter, not 451 lines.
            # flush=True forces it to display immediately (print normally
            # buffers, which would make the counter lag or appear all at once).
            print(f"\r[run_eval] {i}/{total} samples done", end="", flush=True)
    finally:
        print()  # newline to close off the \r progress line cleanly
        # detach in a finally block so the model is always restored to its
        # original state, even if generation throws partway through.
        method.detach(loaded_model.model)

    # --- Aggregate metrics --------------------------------------------------
    # Split predictions by answer_type so each gets the right metric.
    closed = [p for p in predictions if p["answer_type"] == "closed"]
    open_ = [p for p in predictions if p["answer_type"] == "open"]

    closed_acc = closed_ended_accuracy(
        [p["prediction"] for p in closed],
        [p["ground_truth"] for p in closed],
    ) if closed else None

    open_acc = open_ended_recall(
        [p["prediction"] for p in open_],
        [p["ground_truth"] for p in open_],
    ) if open_ else None

    # Overall accuracy: closed questions scored by exact match, open by
    # recall, combined as a sample-weighted average. Reported alongside the
    # per-type numbers, not instead of them -- the split matters for analysis.
    n_total = len(predictions)
    overall = None
    if n_total > 0:
        closed_sum = (closed_acc * len(closed)) if closed_acc is not None else 0.0
        open_sum = (open_acc * len(open_)) if open_acc is not None else 0.0
        overall = (closed_sum + open_sum) / n_total

    # Latency summary can fail if there were fewer samples than the warmup
    # count (e.g. a --max-samples 3 smoke test). Handle that gracefully
    # rather than crashing after a full eval run.
    try:
        mean_latency_ms, std_latency_ms = tracker.summary()
    except RuntimeError:
        mean_latency_ms, std_latency_ms = None, None

    peak_mem = peak_gpu_memory_gb()

    metrics = {
        "n_total": n_total,
        "n_closed": len(closed),
        "n_open": len(open_),
        "closed_accuracy": closed_acc,
        "open_recall": open_acc,
        "overall_accuracy": overall,
        "mean_latency_ms": mean_latency_ms,
        "std_latency_ms": std_latency_ms,
        "peak_gpu_memory_gb": peak_mem,
    }

    config = {
        "method": method.name,
        "dataset": dataset.name,
        "split": dataset.split,
        "max_samples": dataset.max_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "conv_mode": loaded_model.conv_mode,
    }

    return {
        "metrics": metrics,
        "predictions": predictions,
        "config": config,
    }
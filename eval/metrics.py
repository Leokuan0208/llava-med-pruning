"""Metrics for evaluating medical VQA outputs.

Two categories of metrics:

1. Accuracy metrics: closed-ended (exact match), open-ended (token recall).
   These run on the predictions after evaluation.

2. Performance metrics: latency, peak GPU memory, theoretical FLOPs.
   These are measured during evaluation; FLOPs is per-query, latency
   and memory are aggregated across the run.

The accuracy formulas for medical VQA follow the conventions in the
LLaVA-Med paper and the broader Med-VQA literature: closed questions are
scored by case-insensitive exact match; open questions are scored by
token-level recall (fraction of ground-truth tokens that appear in the
prediction). Open-recall is the standard choice over F1 because medical
answers are often short and a strict F1 over-penalizes verbose-but-correct
responses.
"""

import re
import statistics
import time
from typing import List, Tuple

import torch


# ---------------------------------------------------------------------------
# Text normalization helper
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace, and drop trailing punctuation.

    Used so that "Yes.", " yes", and "YES" all compare equal. This is a
    deliberately light touch: we do NOT strip internal punctuation or do
    stemming, because that can change medical meaning.
    """
    text = text.strip().lower()
    # Remove leading/trailing punctuation but keep internal characters intact.
    text = text.strip(".,!?;:\"'()[]{} ")
    return text


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens.

    re.findall(r"\\w+", ...) grabs runs of alphanumeric characters, which
    discards punctuation cleanly. "the lung's edge." -> ["the","lung","s","edge"]
    """
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def closed_ended_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Case-insensitive exact-match accuracy for closed (yes/no, MC) questions.

    A prediction counts as correct if the normalized ground-truth answer
    appears as a whole word in the normalized prediction. This is more
    lenient than strict string equality, which matters because the model
    often answers "Yes, there is evidence of..." instead of just "yes".

    Args:
        predictions: Model predictions (one string per question).
        ground_truths: Ground-truth answers (one string per question).

    Returns:
        Fraction in [0, 1]. Returns 0.0 if the input lists are empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths."
        )
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = _normalize(pred)
        gt_norm = _normalize(gt)
        # Whole-word check: build a set of the prediction's tokens and test
        # membership, so "no" does not match inside "nodule".
        pred_tokens = set(_tokenize(pred_norm))
        if gt_norm in pred_tokens or gt_norm == pred_norm:
            correct += 1
    return correct / len(predictions)


def open_ended_recall(predictions: List[str], ground_truths: List[str]) -> float:
    """Token-level recall for open-ended questions.

    For each (pred, gt) pair: tokenize both (lowercase, split on whitespace
    and punctuation), then compute |pred_tokens ∩ gt_tokens| / |gt_tokens|.
    Average over all pairs.

    Args:
        predictions: Model predictions (one string per question).
        ground_truths: Ground-truth answers (one string per question).

    Returns:
        Mean recall in [0, 1]. Returns 0.0 if the input lists are empty.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths."
        )
    if not predictions:
        return 0.0

    recalls = []
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = set(_tokenize(pred))
        gt_tokens = set(_tokenize(gt))
        if not gt_tokens:
            # Degenerate gold answer: count it correct only if pred is also empty.
            recalls.append(1.0 if not pred_tokens else 0.0)
            continue
        overlap = pred_tokens & gt_tokens
        recalls.append(len(overlap) / len(gt_tokens))
    return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
class LatencyTracker:
    """Measures wall-clock generation latency with proper CUDA synchronization.

    Usage:
        tracker = LatencyTracker(warmup=5)
        for question in dataset:
            tracker.start()
            # ... run model.generate(...)
            tracker.stop()
        mean_ms, std_ms = tracker.summary()

    Warmup runs are excluded from the summary. CUDA synchronization
    happens at start() and stop() to ensure measurements aren't
    confounded by async kernel launches.
    """

    def __init__(self, warmup: int = 5):
        self.warmup = warmup
        self.timings: List[float] = []   # every recorded duration, in seconds
        self._t0: float = None           # timestamp set by start(), cleared by stop()

    def start(self) -> None:
        """Synchronize the GPU, then record the start timestamp."""
        if torch.cuda.is_available():
            # Block until all previously-queued GPU work is done, so the
            # timer starts from a known-idle state.
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()   # perf_counter is the highest-resolution clock

    def stop(self) -> None:
        """Synchronize the GPU again, then record the elapsed duration."""
        if self._t0 is None:
            raise RuntimeError("stop() called before start().")
        if torch.cuda.is_available():
            # Block until the generate() work has actually finished on the GPU.
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0
        self.timings.append(elapsed)
        self._t0 = None

    def summary(self) -> Tuple[float, float]:
        """Return (mean_ms, std_ms) over non-warmup runs.

        Raises:
            RuntimeError: if there are not enough timings to report
                anything after discarding warmup runs.
        """
        usable = self.timings[self.warmup:]
        if not usable:
            raise RuntimeError(
                f"No timings left after discarding {self.warmup} warmup runs "
                f"(only {len(self.timings)} total recorded)."
            )
        # Convert seconds -> milliseconds for a more readable number.
        usable_ms = [t * 1000.0 for t in usable]
        mean_ms = statistics.fmean(usable_ms)
        # stdev needs at least 2 data points; with 1, report 0.0.
        std_ms = statistics.stdev(usable_ms) if len(usable_ms) > 1 else 0.0
        return mean_ms, std_ms


def peak_gpu_memory_gb(device: int = 0) -> float:
    """Return peak GPU memory in GiB since the last reset.

    Wraps torch.cuda.max_memory_allocated. Call reset_peak_memory
    before the run, then call this after.
    """
    if not torch.cuda.is_available():
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 3)   # bytes -> GiB


def reset_peak_memory(device: int = 0) -> None:
    """Reset the peak-memory counter so subsequent peak readings are clean."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
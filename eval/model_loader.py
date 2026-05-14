"""Thin wrapper around llava.model.builder.load_pretrained_model.

Centralizing model loading here means:

1. The runner doesn't need to import LLaVA internals directly.
2. If we add support for other base models later (MedGemma, etc.), only
   this file changes.
3. We can stash the auxiliary objects (tokenizer, image_processor) in one
   place and pass a single LoadedModel object to downstream code.
"""

import os
from dataclasses import dataclass
from typing import Any

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


@dataclass
class LoadedModel:
    """Everything the runner needs to generate from a VLM.

    Bundling these together means the runner has one object to pass
    around instead of four.
    """
    model: Any           # The HuggingFace model object
    tokenizer: Any       # Tokenizer for prompt encoding
    image_processor: Any # Vision-tower preprocessor (CLIP for v1.5)
    context_len: int     # Max sequence length the model supports
    conv_mode: str       # Which conversation template to use


# Maps a substring of the model directory name to the conversation template
# that model expects. v1.5 of LLaVA-Med is Mistral-based and uses the
# "mistral_instruct" template -- the same one used for working CLI inference.
_CONV_MODE_BY_MODEL = {
    "mistral": "mistral_instruct",
    "llama": "llava_v1",
}


def _infer_conv_mode(model_name: str) -> str:
    """Pick the conversation template from the model directory name.

    Args:
        model_name: The model name string (e.g. "llava-med-v1.5-mistral-7b").

    Returns:
        A conv-template identifier understood by llava.conversation.

    Raises:
        ValueError: if no known keyword is found, so we fail loudly rather
            than silently using the wrong prompt format.
    """
    lowered = model_name.lower()
    for keyword, conv_mode in _CONV_MODE_BY_MODEL.items():
        if keyword in lowered:
            return conv_mode
    raise ValueError(
        f"Could not infer a conversation mode from model name '{model_name}'. "
        f"Known keywords: {list(_CONV_MODE_BY_MODEL)}. "
        f"Add an entry to _CONV_MODE_BY_MODEL if this is a new model family."
    )


def load_llava_med(model_path: str, device: str = "cuda") -> LoadedModel:
    """Load LLaVA-Med v1.5 from the given path.

    Args:
        model_path: Filesystem path to a HuggingFace-format model dir,
            e.g. /data/dan/weights/llava-med-v1.5-mistral-7b.
        device: Where to place the model. Almost always "cuda".

    Returns:
        LoadedModel bundle.

    Raises:
        FileNotFoundError: if model_path does not exist on disk.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path does not exist or is not a directory: {model_path}"
        )

    # get_model_name_from_path turns "/data/dan/weights/llava-med-v1.5-mistral-7b"
    # into "llava-med-v1.5-mistral-7b". load_pretrained_model uses this name
    # internally to decide how to construct the model.
    model_name = get_model_name_from_path(model_path)

    # The core call. LLaVA-Med's loader returns a 4-tuple in this fixed order.
    # model_base=None       : v1.5 ships as full merged weights, no delta to merge.
    # device_map=device     : pin the whole model to one device explicitly.
    #                         The loader's default is 'auto', which lets accelerate
    #                         decide placement -- harmless on a single A100, but in
    #                         research code implicit device placement is worth
    #                         making explicit so runs are predictable.
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map=device,
        device=device,
    )

    conv_mode = _infer_conv_mode(model_name)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        context_len=context_len,
        conv_mode=conv_mode,
    )
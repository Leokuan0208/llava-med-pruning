"""PathVQA dataset loader.

PathVQA (He et al., 2020): ~32,000 question-answer pairs over ~4,000
pathology images extracted from textbooks and digital libraries.
Heavily weighted toward histology and pathology, complementing the
radiology focus of VQA-RAD and SLAKE.

Source: https://github.com/UCSD-AI4H/PathVQA
Expected directory layout under root:
    root/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── qas/
        ├── train_qa.pkl
        ├── val_qa.pkl
        └── test_qa.pkl

PathVQA is distributed as pickle files. We'll deserialize them in this
loader once the dataset is in place.
"""

from pathlib import Path
from typing import Iterator

from .base import MedVQADataset, VQASample


class PathVQADataset(MedVQADataset):
    """Loader for PathVQA. Implementation pending dataset download."""

    @property
    def name(self) -> str:
        return "path_vqa"

    def __iter__(self) -> Iterator[VQASample]:
        raise NotImplementedError(
            "PathVQA loader is a stub. Will be implemented after the "
            "dataset is downloaded and we can inspect the pickle layout."
        )

    def __len__(self) -> int:
        raise NotImplementedError

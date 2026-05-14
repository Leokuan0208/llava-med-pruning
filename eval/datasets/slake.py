"""SLAKE dataset loader.

SLAKE (Liu et al., 2021): ~14,000 question-answer pairs over 642 medical
images. Bilingual (English + Chinese; we use English only). Multi-organ
coverage. Provides closed and open-ended questions plus semantic labels.

Source: https://www.med-vqa.com/slake/
Expected directory layout under root:
    root/
    ├── imgs/
    ├── train.json
    ├── validate.json
    └── test.json

The exact layout depends on how the dataset is downloaded. This loader
will need to be filled in once we inspect the actual file structure.
"""

from pathlib import Path
from typing import Iterator

from .base import MedVQADataset, VQASample


class SlakeDataset(MedVQADataset):
    """Loader for SLAKE. Implementation pending dataset download."""

    @property
    def name(self) -> str:
        return "slake"

    def __iter__(self) -> Iterator[VQASample]:
        raise NotImplementedError(
            "SLAKE loader is a stub. Will be implemented after the "
            "dataset is downloaded and we can inspect the file layout."
        )

    def __len__(self) -> int:
        raise NotImplementedError

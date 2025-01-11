from pathlib import Path
from dataclasses import dataclass
from omegaconf import ListConfig
from typing import List


@dataclass
class LabeledDataset:
    data_dir: Path
    # Ordered list of paths, one for each view in the config.
    # For single-view models, this is a list of size 1.
    csv_files: List[Path]
    train_prob: float = 1.0
    val_prob: float = 0.0
    train_frames: int = 1

    # Memory of whether csv_file was string or list.
    _csv_file_was_list: bool = False

    @staticmethod
    def from_cfg(cfg):
        _csv_file_was_list = isinstance(cfg.data.csv_file, ListConfig)
        csv_files = cfg.data.csv_file if _csv_file_was_list else [cfg.data.csv_file]
        return LabeledDataset(
            data_dir=Path(cfg.data.data_dir),
            csv_files=[Path(f) for f in csv_files],
            train_prob=cfg.training.train_prob,
            val_prob=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
            _csv_file_was_list=_csv_file_was_list,
        )

    def to_partial_cfg(self):
        return {
            "data": {
                "data_dir": str(self.data_dir),
                "csv_file": (
                    [str(p) for p in self.csv_files]
                    if self._csv_file_was_list
                    else str(self.csv_files[0])
                ),
            },
            "training": {
                "train_prob": self.train_prob,
                "val_prob": self.val_prob,
                "train_frames": self.train_frames,
            },
        }

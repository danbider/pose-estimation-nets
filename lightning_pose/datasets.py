from pathlib import Path
from dataclasses import dataclass


@dataclass
class LabeledDataset:
    data_dir: Path
    csv_file: Path
    train_prob: float = 1.0
    val_prob: float = 0.0
    train_frames: int = 1

    @staticmethod
    def from_cfg(cfg):
        return LabeledDataset(
            data_dir=Path(cfg.data.data_dir),
            csv_file=Path(cfg.data.csv_file),
            train_prob=cfg.training.train_prob,
            val_prob=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
        )

    def to_partial_cfg(self):
        return {
            "data": {
                "data_dir": str(self.data_dir),
                "csv_file": str(self.csv_file),
            },
            "training": {
                "train_prob": self.train_prob,
                "val_prob": self.val_prob,
                "train_frames": self.train_frames,
            },
        }

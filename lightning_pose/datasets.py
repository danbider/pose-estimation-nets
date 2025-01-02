from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class LabeledDataset:
    data_dir: Path
    csv_file: Path
    keypoint_names: List[str]
    train_prob: float = 1.0
    val_prob: float = 0.0
    train_frames: int = 1

    @staticmethod
    def from_cfg(cfg):
        return LabeledDataset(
            data_dir=cfg.data.data_dir,
            csv_file=cfg.data.csv_file,
            keypoint_names=cfg.data.keypoint_names,
            train_prob=cfg.training.train_prob,
            val_prob=cfg.training.val_prob,
            train_frames=cfg.training.train_frames,
        )

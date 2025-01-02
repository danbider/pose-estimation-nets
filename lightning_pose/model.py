from typing import TypedDict, Optional

import pandas as pd
from pathlib import Path

from lightning.pytorch import LightningModule
from omegaconf import DictConfig

from lightning_pose.datasets import LabeledFrameDataset
from lightning_pose.model_config import ModelConfig
from lightning_pose.models import ALLOWED_MODELS
from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.predictions import load_model_from_checkpoint

__all__ = ["Model"]

class Model:
    model_dir: Path
    cfg: ModelConfig
    model: LightningModule

    @staticmethod
    def from_dir(model_dir: str | Path):
        model_dir = Path(model_dir)
        cfg = ModelConfig.from_yaml_file(model_dir / "config.yaml")
        return Model(model_dir, cfg)

    def __init__(self, model_dir: str | Path, config: ModelConfig):
        self.model_dir = Path(model_dir)
        self.config = config

    @property
    def cfg(self):
        return self.config.cfg

    def _load(self):
        ckpt_file = ckpt_path_from_base_path(
            base_path=str(self.model_dir), model_name=self.cfg.model.model_name
        )
        self.model: ALLOWED_MODELS = load_model_from_checkpoint(
            cfg=self.cfg, ckpt_file=ckpt_file, eval=True, skip_data_module=True,
        )


    #############
    # Prediction
    #############

    class PredictionResult(TypedDict):
        predictions: pd.DataFrame
        metrics: pd.DataFrame
        output_locations: dict

    UNSPECIFIED = ''

    def predict_dataset(self,
                        dataset: LabeledDataset,
                        prediction_output_path: Optional[str] = UNSPECIFIED,
                        compute_metrics: bool=True,
                        metric_output_path: str = UNSPECIFIED,
                        ) -> PredictionResult:
        """
        Run prediction on a labeled dataset.
        Compute metrics.
        Save outputs.

        Args:
            dataset:

        Returns:

        """
        self._load()

        preds_file = prediction_output_path

        if prediction_output_path is None:
            raise NotImplementedError("predicting without saving file is not yet implemented.")

        if prediction_output_path == self.UNSPECIFIED:
            if self.config.is_single_view():
                preds_file = "predictions.csv"
            else:
                # TODO implement format string support in predict_dataset function.
                preds_file = "predictions_{view_name}.csv"

        from lightning_pose.utils.predictions import predict_dataset
        data_module = _build_datamodule_pred(self.cfg)
        df = predict_dataset(self.cfg, data_module, model=self.model, preds_file=preds_file)

        #if compute_metrics:
        #    compute_metrics()

        return self.PredictionResult(predictions = df)


def _build_datamodule_pred(cfg: DictConfig):
    # Legacy predict_dataset fn requires a datamodule. TODO move to predict_dataset.
    from lightning_pose.utils.scripts import (
        get_dataset,
        get_data_module,
        get_imgaug_transform,
        )
    import copy
    cfg_pred = copy.deepcopy(cfg)
    cfg_pred.training.imgaug = "default"
    imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
    dataset_pred = get_dataset(
        cfg=cfg_pred, data_dir=cfg_pred.data.data_dir, imgaug_transform=imgaug_transform_pred
    )
    data_module_pred = get_data_module(cfg=cfg_pred, dataset=dataset_pred, video_dir=cfg_pred.data.video_dir)
    data_module_pred.setup()

    return data_module_pred
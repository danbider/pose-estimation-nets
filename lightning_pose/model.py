import copy
from typing import TypedDict, Optional

import pandas as pd
from pathlib import Path

from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf

from lightning_pose.datasets import LabeledDataset
from lightning_pose.model_config import ModelConfig
from lightning_pose.models import ALLOWED_MODELS
from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.predictions import (
    load_model_from_checkpoint,
    export_predictions_and_labeled_video,
)
from lightning_pose.utils.scripts import compute_metrics as compute_metrics_fn

__all__ = ["Model"]


class Model:
    model_dir: Path
    config: ModelConfig
    model: Optional[ALLOWED_MODELS] = None

    @staticmethod
    def from_dir(model_dir: str | Path):
        model_dir = Path(model_dir)
        config = ModelConfig.from_yaml_file(model_dir / "config.yaml")
        return Model(model_dir, config)

    def __init__(self, model_dir: str | Path, config: ModelConfig):
        self.model_dir = Path(model_dir)
        self.config = config

    @property
    def cfg(self):
        return self.config.cfg

    def _load(self):
        if self.model is None:
            ckpt_file = ckpt_path_from_base_path(
                base_path=str(self.model_dir), model_name=self.cfg.model.model_name
            )
            self.model = load_model_from_checkpoint(
                cfg=self.cfg,
                ckpt_file=ckpt_file,
                eval=True,
                skip_data_module=True,
            )

    #############
    # Prediction
    #############

    class PredictionResult(TypedDict):
        predictions: pd.DataFrame
        metrics: pd.DataFrame
        output_locations: dict


    """
    TODO: should we default to this being in the model directory?
    TODO: should we put the files in folders for better organization?
    TODO: and to resolve parsing ambiguity with prediction_new_{view_name}?
    TODO: Ask Matt.
    """
    def predict_frames(
        self,
        dataset: LabeledDataset,
        prediction_output_path: Optional[str] = None,
        compute_metrics: bool = True,
        metric_output_path: Optional[str] = None,
    ) -> PredictionResult:
        """Predicts on a labeled dataset (or unlabeled frames, not yet supported).
        Args:
            dataset (LabeledDataset): The labeled dataset to predict on.
            prediction_output_path (Optional[str], optional): The path to save the
                                                               predictions to. If None,
                                                               predictions are not saved.
            compute_metrics (bool, optional): Whether to compute metrics on the
                                              predictions. Defaults to True.
            metric_output_path (str, optional): The path to save the metrics to.
                                                If unspecified and compute_metrics is True,
                                                metrics are not saved.
                                                Defaults to UNSPECIFIED.

        Returns:
            PredictionResult: A PredictionResult object containing the predictions
                              and metrics.
        """
        self._load()

        preds_file = prediction_output_path

        if prediction_output_path is None:
            raise NotImplementedError(
                "predicting without saving file is not yet implemented."
            )

        from lightning_pose.utils.predictions import predict_dataset

        cfg_pred = OmegaConf.merge(self.cfg, dataset.to_partial_cfg())
        data_module = _build_datamodule_pred(cfg_pred)
        df = predict_dataset(
            cfg_pred, data_module, model=self.model, preds_file=preds_file
        )

        if compute_metrics:
            # TODO: Fix Multiview logic for preds_file. compute_metrics currently inputs a list of preds_files.
            # Make it accept a pattern instead.
            compute_metrics_fn(cfg_pred, preds_file, data_module)

        # TODO: Generate detector outputs.

        return self.PredictionResult(predictions=df)

    def predict_video_file(
        self,
        video_file: str | Path,
        prediction_output_path: Optional[str] = None,
        compute_metrics: bool = True,
        metric_output_path: Optional[str] = None,
        generate_labeled_video: bool = False,
    ) -> PredictionResult:
        self._load()
        video_file = Path(video_file)

        prediction_csv_file = self.model_dir / "video_preds" / f"{video_file.stem}.csv"

        labeled_mp4_file = None
        if generate_labeled_video:
            labeled_mp4_file = (
                self.model_dir / "labeled_videos" / f"{video_file.stem}_labeled.mp4"
            )

        if self.config.cfg.eval.get("predict_vids_after_training_save_heatmaps", False):
            raise NotImplementedError(
                "Implement this after cleaning up _predict_frames: "
                "Set a flag on the model to return heatmaps. "
                "Use trainer.predict instead of side-stepping it."
            )
        df = export_predictions_and_labeled_video(
            video_file=str(video_file),
            cfg=self.config.cfg,
            prediction_csv_file=str(prediction_csv_file),
            labeled_mp4_file=labeled_mp4_file,
            model=self.model,
        )

        # This is only needed for computing PCA metrics.
        # TODO push this down to compute metrics?
        data_module = _build_datamodule_pred(self.cfg)
        if compute_metrics:
           compute_metrics_fn(self.cfg, str(prediction_csv_file), data_module)

        return self.PredictionResult(predictions=df)


def _build_datamodule_pred(cfg: DictConfig):
    # Legacy predict_dataset fn requires a datamodule. TODO move to legacy predict_dataset.
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
        cfg=cfg_pred,
        data_dir=cfg_pred.data.data_dir,
        imgaug_transform=imgaug_transform_pred,
    )
    data_module_pred = get_data_module(
        cfg=cfg_pred, dataset=dataset_pred, video_dir=cfg_pred.data.video_dir
    )
    data_module_pred.setup()

    return data_module_pred

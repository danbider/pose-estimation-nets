from omegaconf import OmegaConf, DictConfig

__all__ = ["ModelConfig"]

class ModelConfig:

    @staticmethod
    def from_yaml_file(filepath):
        return ModelConfig(OmegaConf.load(filepath))

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def is_single_view(self):
        return not self.is_multi_view()

    def is_multi_view(self):
        return self.cfg.data.get('view_names') is not None and len(self.cfg.data.view_names) > 1


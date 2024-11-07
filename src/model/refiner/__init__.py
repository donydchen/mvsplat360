from omegaconf import OmegaConf
from .svd_diffuser import SVDDiffusion, SVDDiffusionCfg
from sgm.util import get_obj_from_str


RefinerCfg = SVDDiffusionCfg


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_refiner(cfg: RefinerCfg):
    config = OmegaConf.load(cfg.config_path)
    model = instantiate_from_config(config.model, cfg=cfg)
    return model

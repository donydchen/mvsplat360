import os
from pathlib import Path
import sys
import json
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.refiner import get_refiner
    from src.model.model_wrapper import ModelWrapper


def yellow(text: str) -> str:
    return f"{Fore.YELLOW}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    # overwrite cfgs
    warning_infos = []
    if cfg_dict["checkpointing"]["load"]:
        cfg_dict["checkpointing"]["pretrained_model"] = None
        warning_infos.append(yellow("Resume from checkpoint, ignore encoder init."))
        if "refiner" in cfg_dict["model"] and cfg_dict["model"]["refiner"].get("ckpt_path", None):
            cfg_dict["model"]["refiner"]["ckpt_path"] = None
            warning_infos.append(yellow("Resume from checkpoint, ignore SD init."))

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    warning_infos.append(yellow(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        precision=str(cfg.trainer.precision),
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy=(
            "ddp"
            if torch.cuda.device_count() > 1 and cfg.trainer.num_nodes == 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "refiner": get_refiner(cfg.model.refiner) if cfg.use_diff_refinement else None,
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights but no optimizer state
        print(f"Loading full model weights from {checkpoint_path}")
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path,
            **model_kwargs,
            strict=True,
            map_location="cpu",
        )
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    # log running info
    if trainer.global_rank == 0:
        if cfg.mode == "train":
            saved_root = output_dir
        else:
            saved_root = cfg.test.output_path / cfg.wandb["name"]
            os.makedirs(saved_root, exist_ok=True)
        with open(saved_root / "runs.sh", "a+") as f:
            f.write(f"python {' '.join(sys.argv)}\n")
        with open(saved_root / "configs.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg_dict, resolve=True), f, indent=2)
        for warning_info in warning_infos:
            print(warning_info)

    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path if cfg.checkpointing.resume else None,
        )
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()

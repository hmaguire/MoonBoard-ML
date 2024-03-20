from multiprocessing import freeze_support
from pathlib import Path

from lightning.pytorch.cli import LightningCLI
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
# Hide lines below until Lab 04
from grade_predictor import callbacks as cb

# Hide lines above until Lab 04
from grade_predictor import lit_models
from lightning.pytorch.tuner.tuning import Tuner

from jsonargparse import lazy_instance

@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


class MyLightningCLI(LightningCLI):
    # In order to ensure reproducible experiments, we must set random seeds.
    np.random.seed(42)
    torch.manual_seed(42)

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--wandb",
            action="store_true",
            default=False,
            help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
        )

        parser.add_argument(
            "--profile",
            action="store_true",
            default=False,
            help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
        )
        parser.add_argument(
            "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
        )

        parser.add_lightning_class_args(EarlyStopping, "stop_early")
        parser.set_defaults({"stop_early.monitor": "validation/loss",
                             "stop_early.mode": "min",
                             "stop_early.patience": 0})



        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
        parser.set_defaults({"checkpoint.save_top_k": 5,
                             "checkpoint.filename": filename_format,
                             "checkpoint.auto_insert_metric_name": False,
                             "checkpoint.dirpath": "trainer.logger.log_dir"
                             })
        parser.link_arguments("stop_early.monitor", "checkpoint.monitor")
        parser.link_arguments("stop_early.mode", "checkpoint.mode")
        parser.link_arguments("checkpoint.every_n_epochs", "trainer.check_val_every_n_epoch")

        parser.link_arguments("data", "model.init_args.data_config",
                              compute_fn=lambda data: data.config(),
                              apply_on="instantiate")

        parser.add_lightning_class_args(lit_models.BaseLitModel, "lit_model")



    def before_fit(self):
        args = self.config.fit

        lit_model_class = lit_models.BaseLitModel

        if args.load_checkpoint is not None:
            lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=self.model)
        else:
            lit_model = lit_model_class(model=self.model)

        self.model = lit_model
        Tuner(self.trainer).lr_find(model=self.model, datamodule=self.datamodule)

        log_dir = Path("training") / "logs"
        _ensure_logging_dir(log_dir)
        self.trainer.logger = TensorBoardLogger(log_dir)
        experiment_dir = self.trainer.logger.log_dir

        goldstar_metric = "validation/loss"
        filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
        checkpoint_callback = ModelCheckpoint(
            save_top_k=5,
            filename=filename_format,
            monitor=goldstar_metric,
            mode="min",
            auto_insert_metric_name=False,
            dirpath=experiment_dir,
            every_n_epochs=self.trainer.check_val_every_n_epoch,
        )

        summary_callback = ModelSummary(max_depth=2)

        self.trainer.callbacks = [summary_callback, checkpoint_callback]

        if args.wandb:
            self.trainer.logger = WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train",
                                 project="MB2016-grade-predictor")
            # self.trainer.logger.watch(args.model, log_freq=max(100, 10))
            # self.trainer.logger.log_hyperparams(vars(args))
            experiment_dir = self.trainer.logger.experiment.dir

        self.trainer.callbacks += [cb.ModelSizeLogger(), cb.LearningRateMonitor()]
        # Hide lines above until Lab 04
        # if args.stop_early:
        #     early_stopping_callback = EarlyStopping(
        #         monitor="validation/loss", mode="min", patience=args.stop_early
        #     )
        #     self.trainer.callbacks.append(early_stopping_callback)




        # lit_model and model and data
        # call backs


        # if args.load_checkpoint is not None:
        #     lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=args.model)
        # else:
        #     lit_model = lit_model_class(model=model)

    def after_fit(self):
        pass

if __name__ == '__main__':
    # freeze_support()
    cli = MyLightningCLI()

from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import DataModule
from models import SequenceClassificationModel


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = DataModule(
        cfg.datasets.task_name,
        cfg.model.model_name_or_path,
        cfg.trainer.max_seq_length,
        cfg.trainer.train_batch_size,
        cfg.trainer.eval_batch_size,
    )
    model = SequenceClassificationModel(
        cfg.model.model_name_or_path,
        dm.num_labels,
        train_batch_size=cfg.trainer.train_batch_size,
        eval_batch_size=cfg.trainer.eval_batch_size,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.trainer.cpkt_path,
        filename="{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        auto_insert_metric_name=True,
        save_top_k=-1,
    )
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        # tpu_cores=1,
        gpus=1,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
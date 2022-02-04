from pytorch_lightning import LightningModule
from torch import logit
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
import datasets
import torch
from sklearn.metrics import accuracy_score


class SequenceClassificationModel(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        weight_decay: float = 1.0,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        else:
            preds = logits.squeeze()

        labels = batch["labels"]

        validation_output = {"loss": val_loss, "preds": preds, "labels": labels}

        return validation_output

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy_score(labels, preds), prog_bar=True)

    def setup(self, stage=None):
        if stage != "fit":
            return
        train_loader = self.trainer.datamodule.train_dataloader()

        tb_size = self.hparams.train_batch_size * max(1, int(self.trainer.gpus or 1))
        # accumulate_grad_batches는 몇 번의 배치동안 누적해서 계산할 것인지
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(LightningDataModule):

    task_text_field_map = {
        "ynat": ["title"],
        "sts": ["sentence1", "sentence2"],
        "nli": ["premise", "hypothesis"],
        "ner": ["tokens"],
        # "re": ["sentence"], #TODO: subject_entity, object_entity 처리
        # "dp": ["sentence"],
        # "mrc": [""],
        # "wos": [""],
    }

    task_label_field_map = {
        "ynat": "label",
        "sts": "label",
        "nli": "label",
        "ner": "ner_tags",
        # "re": ["sentence"], #TODO: subject_entity, object_entity 처리
        # "dp": ["sentence"],
        # "mrc": [""],
        # "wos": [""],
    }
    
    task_num_labels = {
        "ynat": 7,
    }

    def __init__(
        self,
        task_name: str,
        model_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
    ):
        super().__init__()
        self.task_name = task_name
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.task_num_labels[task_name] # 하드코딩 말고 데이터셋에서 구해올 수는 없나?

        # self.prepare_data()

    def prepare_data(self):
        self.dataset = load_dataset("klue", self.task_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return
        
        for split in self.dataset.keys():

            # if split == "train":
            #     self.num_labels = (
            #         self.dataset[split]
            #         .info.features[self.task_label_field_map[self.task_name]]
            #         .num_classes
            #     )

            # TODO: remove_columns 안하면 어떻게 되나? 메모리 더 쓰나?
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                # remove_columns=self.dataset[split].column_names,
            )

            # TODO: task마다 columns가 다를 수 있음. 변경 필요.
            self.dataset[split].set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
            )

    def convert_to_features(self, example_batch):
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer(
            texts_or_text_pairs,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )
        # TODO: ner은 label이 아님.
        features["labels"] = example_batch["label"]
        return features

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)

from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.data.processors.squad import (
    squad_convert_examples_to_features,
    SquadExample,
)


class DataModule(LightningDataModule):

    task_text_field_map = {
        "ynat": ["title"],
        "mrc": ["question", "context"],
    }

    task_label_field_map = {
        "ynat": "label",
        "mrc": "answers",
    }

    task_num_labels = {
        "ynat": 7,
        "mrc": 0,
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
        self.label_fields = self.task_label_field_map[task_name]
        self.num_labels = self.task_num_labels[task_name]  # 하드코딩 말고 데이터셋에서 구해올 수는 없나?

    def prepare_data(self):
        self.dataset = load_dataset("klue", self.task_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return

        for split in self.dataset.keys():

            # TODO: remove_columns 안하면 어떻게 되나? 메모리 더 쓰나?
            self.dataset[split] = self.dataset[split].train_test_split(test_size=0.1)['test']
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

        if self.task_name == "ynat":
            features = self.tokenizer(
                texts_or_text_pairs,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
            )
            features[self.label_fields] = example_batch[self.label_fields]

        elif self.task_name == "mrc":
            mode = "train"
            examples = self._create_examples(example_batch, mode)
            features = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=128,
                max_query_length=64,
                is_training=True,
                # return_dataset="pt",
            )
            features["start_positions"] = None
            features["end_positions"] = None

        return features

    def _create_examples(self, input_data, set_type):
        # features: ['title', 'context', 'news_category', 'source', 'guid', 'is_impossible', 'question_type', 'question', 'answers'],
        is_training = set_type == "train"
        examples = []
        
        for idx in tqdm(range(len(input_data['title']))):
            title = input_data["title"][idx]
            context_text = input_data["context"][idx]
            qas_id = input_data["guid"][idx]
            question_text = input_data["question"][idx]
            answers = []
            
            is_impossible = input_data["is_impossible"][idx]
            if not is_impossible:
                if is_training:
                    answer = input_data["answers"][idx]
                    answer_text = answer["text"][0]
                    start_position_character = answer["answer_start"][0]
                else:
                    answers = input_data["answers"][idx]
            
            example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
            )
            examples.append(example)
        return examples

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)

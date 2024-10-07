from typing import List, Dict, Any
import zipfile, glob, json
import pandas as pd
from collections.abc import Mapping
import numpy as np

import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
from datasets import load_dataset
from utils import MyDataset


############### From local zip file ###############
def check_data_on_wd(data_dir, zip_path, type_="train") -> List[str]:
    """학습 및 테스트 데이터가 적절한 디렉터리에 있는지 검사하고 없다면,
    zip 파일을 압축 해제하는 함수입니다."""
    try:
        zip = zipfile.ZipFile(zip_path)
        zip.extractall("/root/exp")
    except:
        print(".zip파일이 존재하지 않습니다.")
    return glob.glob(f"{data_dir}/*{type_}.jsonl")


def jsonl_to_pandas(
    data_dir, zip_path, type_="train", submission=False
) -> pd.DataFrame:
    "data_dir에 존재하는 josnl 파일을 불러와 데이터 프레임화 하는 함수입니다."
    # 파일 이름이 valid가 아니라 dev로 되어 있다.
    if type_ == "valid":
        type_ = "dev"
    data_path = glob.glob(f"{data_dir}/*{type_}.jsonl")
    if not data_path:
        data_path = check_data_on_wd(data_dir, zip_path, type_)

    temp_dicts = []
    with open(data_path[0], "r", encoding="utf-8") as f:
        for line in f:
            temp_dicts.append(json.loads(line))

    if submission:
        return pd.DataFrame(temp_dicts)
    return pd.DataFrame(temp_dicts).drop(["id"], axis=1)


class Dataset_v1(Dataset):
    def __init__(self, data: Dict, train=True):
        self.input_ids = data["input_ids"]
        self.token_type_ids = data["token_type_ids"]
        self.attention_mask = data["attention_mask"]
        self.train = train
        if self.train:
            self.label = data["label"]
        self.length = len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        if self.train:
            label = self.label[idx]
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "label": label,
            }
        else:
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }

    def __len__(self):
        return self.length


def get_dataset(config, tokenizer, type_="train", submission=False):
    """zip파일에서 필요한 데이터를 불러와 Trainer가 원하는 형태인
    pytorch Dataset 형태로 바꾸어 주는 과정이 담긴 허브 함수입니다."""

    # config의 test_run 값에 따라서 데이터를 얼마나 불러올지 결정
    if config.test_run:
        flag = 500
    else:
        flag = None
    # jsonl -> pd.DataFrame
    df = jsonl_to_pandas(
        config.dataset_dir, config.zip_path, type_=type_, submission=submission
    )[:flag]
    if submission:
        return df

    # raw data -> tokenized data
    tokenized_data = tokenizer(
        df.input.to_list(),
        max_length=config.max_len,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    # train 및 valid 데이터만 label이 존재
    train = False if type_ == "test" else True
    if train:
        if config.num_labels == 1:
            tokenized_data["label"] = FloatTensor(df["output"].to_numpy()).reshape(
                -1, 1
            )
        else:
            tokenized_data["label"] = LongTensor(df["output"].to_numpy()).reshape(-1, 1)

    # pd.DataFrame -> pytorch Dataset

    dataset = Dataset_v1(tokenized_data, train=train)

    return dataset


############### From huggingface hub ###############
def tokenize_hf_dataset(example, tokenizer, max_len):
    return tokenizer(
        example["input"],
        max_length=max_len,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )


def get_dataset_hf(config, tokenizer, type_="train", submission=False):
    """huggingface hub에서 데이터를 불러와 mapping후 전달하는 과정이 담긴 허브 함수입니다."""

    # huggingface에서 Datasets.Dataset 불러오기
    # config의 test_run 값에 따라서 데이터를 얼마나 불러올지 결정
    if config.test_run:
        data = load_dataset(
            config.dataset_dir, revision=config.dataset_revision, split=type_
        ).select(range(100))
        print("Using TEST RUN DATASET")
    else:
        data = load_dataset(
            config.dataset_dir, revision=config.dataset_revision, split=type_
        )
        print("Using WHOLE DATASET")

    if submission:
        return pd.DataFrame(data)

    if config.dataset_dir == "100suping/malpyeong-hate-speech":
        # mapping
        tokenized_data = data.map(
            tokenize_hf_dataset,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_len": config.max_len},
        )
        # ToDo: 다음에 데이터 셋을 저장할 때, true 값을 output이 아니라 label column에 저장할 것

        tokenized_data = tokenized_data.rename_column("output", "label")
    else:
        tokenized_data = MyDataset(data)

    return tokenized_data


def data_collator_for_label_reshape(features) -> Dict[str, Any]:
    """hf hub의 데이터 셋을 사용함에 따라서, 기존의 label 데이터를 reshape 하는 방식((-1,) -> (-1, 1))의 적용이 어려워 만들어진
    transformers trainer용 data_collator 입니다.
    참조 - https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    """

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.float
        # 원하는 형태로 reshape
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], dtype=dtype
        ).reshape(-1, 1)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def data_collator(examples):
    """Inference시 hf hub의 dataset으로 torch DataLoader를 만들기 위해 필요한 collate_fn입니다."""
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for each in examples:
        input_ids.append(torch.LongTensor(each["input_ids"]))
        attention_mask.append(torch.LongTensor(each["attention_mask"]))
        token_type_ids.append(torch.LongTensor(each["token_type_ids"]))

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    token_type_ids = torch.stack(token_type_ids, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

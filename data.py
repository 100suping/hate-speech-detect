from typing import List, Dict
import zipfile, glob, json
import pandas as pd
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor


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
        flag = -1
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

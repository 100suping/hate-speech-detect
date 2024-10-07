import argparse
import os
# import wandb

from model import do_train

def get_config():
    """argparse를 이용해 사용자에게 하이퍼 파라미터를 입력 받는 함수입니다.
    """
    
    # 여러 정보들이 저장될 디렉터리 생성
    # os.makedirs("hj_code/data" ,exist_ok=True)
    # os.makedirs("hj_code/model" ,exist_ok=True)
    # os.makedirs("hj_code/ckpts" ,exist_ok=True)
    # os.makedirs("hj_code/logs" ,exist_ok=True)

    os.makedirs("./data" ,exist_ok=True)
    os.makedirs("./model" ,exist_ok=True)
    os.makedirs("./ckpts" ,exist_ok=True)
    os.makedirs("./logs" ,exist_ok=True)

    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Hyperparameters",
    )

    parser.add_argument(
        "--project-name",
        default="hate-speech",
        type=str,
    )

    parser.add_argument(
        "--zip-path",
        default='./data/NIKL_AU_2023_v1.0_JSONL.zip',
        type=str,
    )

    parser.add_argument(
        "--dataset-dir",
        default='./data/NIKL_AU_2023_COMPETITION_v1.0',
        type=str,
    )

    parser.add_argument(
        "--model-type",
        default='electra',
        choices=['electra', 'bert', 'roberta'],
        type=str,
    )


    # parser.add_argument(
    #     "--model-name",
    #     default="skt/ko-gpt-trinity-1.2B-v0.5", # gpt 모델은 token_type_ids 가 없음 
    #     choices=["klue/roberta-base", "skt/ko-gpt-trinity-1.2B-v0.5", "beomi/korean-hatespeech-multilabel"],
    #     type=str,
    # )

    parser.add_argument(
        "--model-name",
        default="beomi/korean-hatespeech-multilabel",
        choices=["beomi/korean-hatespeech-multilabel", "matthewburke/korean_sentiment"],
        type=str,
    )

    parser.add_argument(
        "--save-dir",
        default='./model',
        type=str,
    )

    parser.add_argument(
        "--ckpt-dir",
        default='./ckpts',
        type=str,
    )

    parser.add_argument(
        "--logging-dir",
        default="./logs",
        type=str,
    )

    parser.add_argument(
        "--num-labels",
        default=1,
        choices=[1, 2],
        type=int,
    )

    parser.add_argument(
        "--epochs",
        default=15,
        type=int
    )

    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
    )

    parser.add_argument(
        "--max-len",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--warmup-steps",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=8e-5,
        type=float,
    )

    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        # "--fp16",  #맥북에서는 안된다고 하는데 파라미터로는 왜 될까?
        # "--fp16",
        "--bfloat16",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        default=10,
        type=int,
    )
    
    parser.add_argument(
        "--patience",
        default=3,
        type=int
    )
    
    parser.add_argument(
        "--threshold",
        default=0.0,
        type=float
    )

    parser.add_argument(
        "--run-name",
        default='-t',
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    config = parser.parse_args()

    return config

if __name__ == "__main__":
    config = get_config()
    # wandb.init(
    # project=config.project_name,
    # name=config.run_name,
    # )
    do_train(config)
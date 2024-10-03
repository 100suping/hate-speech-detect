import argparse
import os

import wandb

from model import do_train


def get_config():
    """argparse를 이용해 사용자에게 하이퍼 파라미터를 입력 받는 함수입니다."""

    # 여러 정보들이 저장될 디렉터리 생성
    # os.makedirs("/root/exp/data", exist_ok=True)
    # os.makedirs("/root/exp/model", exist_ok=True)
    # os.makedirs("/root/exp/ckpts", exist_ok=True)
    # os.makedirs("/root/exp/logs", exist_ok=True)
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
        default="/root/exp/data/NIKL_AU_2023_v1.0_JSONL.zip",
        type=str,
    )

    parser.add_argument(
        "--dataset-dir",
        default="/root/exp/NIKL_AU_2023_COMPETITION_v1.0",
        type=str,
    )

    parser.add_argument(
        "--model-type",
        default="electra",
        choices=["electra", "bert", "roberta"],
        type=str,
    )

    parser.add_argument(
        "--model-name",
        default="beomi/korean-hatespeech-multilabel",
        choices=["beomi/korean-hatespeech-multilabel", "matthewburke/korean_sentiment"],
        type=str,
    )

    parser.add_argument(
        "--save-dir",
        default="/root/exp/model",
        type=str,
    )

    parser.add_argument(
        "--ckpt-dir",
        default="/root/exp/ckpts",
        type=str,
    )

    parser.add_argument(
        "--logging-dir",
        default="/root/exp/logs",
        type=str,
    )

    parser.add_argument(
        "--num-labels",
        default=1,
        choices=[1, 2],
        type=int,
    )

    parser.add_argument("--epochs", default=10, type=int)

    parser.add_argument(
        "--batch-size",
        default=32,
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
        default=5e-5,
        type=float,
    )

    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        "--fp16",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        default=10,
        type=int,
    )

    parser.add_argument("--patience", default=3, type=int)

    parser.add_argument("--threshold", default=0.0, type=float)

    parser.add_argument(
        "--run-name",
        default="-t",
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--test-run",
        default=0,
        choices=[0, 1],
        type=int,
    )
    
    parser.add_argument(
        "--neftune_noise_alpha",
        type=int,
        default=0.1,
        help="학습 시 임베딩 벡터에 노이즈를 추가하여 성능을 향상시킬 수 있는 허깅페이스 옵션"
    )

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    config = get_config()
    wandb.init(
        project=config.project_name,
        name=config.run_name,
    )
    do_train(config)
    wandb.finish()

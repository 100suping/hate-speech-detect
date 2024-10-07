import argparse
import os
import wandb

from huggingface_hub import login
from dotenv import load_dotenv

from utils import set_experiment_dir
from model import do_pre_train


def get_config():
    """argparse를 이용해 사용자에게 하이퍼 파라미터를 입력 받는 함수입니다."""

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
        "--run-name",
        default="pre-train",
        type=str,
        help="wandb에서 쓰일 run_name",
    )

    parser.add_argument(
        "--dataset-dir",
        default="100suping/korean_unlabeled_web_text",
        type=str,
        help="허깅페이스 허브에 있는 데이터셋 경로 [user_id/repo_name]",
    )

    parser.add_argument(
        "--dataset-revision",
        default="main",
        type=str,
        choices=["main"],
        help="허깅페이스 허브에 있는 데이터셋의 브랜치",
    )

    parser.add_argument(
        "--model-type",
        default="roberta",
        choices=["bert", "roberta"],
        type=str,
    )

    parser.add_argument(
        "--model-name",
        default="klue/roberta-base",
        choices=["klue/roberta-base", "klue/roberta-large", "klue/bert-base"],
        type=str,
    )

    parser.add_argument("--absolute-dir", default="pt", type=str)

    parser.add_argument(
        "--save-dir",
        default="model",
        type=str,
        help="로컬에 모델을 저장할 때, pt/--run-name 아래에 모델과 config가 저장될 디렉터리",
    )

    parser.add_argument(
        "--ckpt-dir",
        default="ckpts",
        type=str,
        help="로컬에 모델 체크포인트를 저장할 시, pt/--run-name 아래에 체크포인트가 저장될 디렉터리",
    )

    parser.add_argument(
        "--logging-dir",
        default="logs",
        type=str,
        help="로컬에 로깅을 진행 할 시, pt/--run-name 아래에 로깅이 저장될 디렉터리",
    )

    parser.add_argument("--epochs", default=50, type=int)

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--max-len",
        default=512,
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

    parser.add_argument(
        "--neftune-noise-alpha",
        type=int,
        default=5,
        help="학습 시 임베딩 벡터에 노이즈를 추가하여 성능을 향상시킬 수 있는 허깅페이스 옵션",
    )

    parser.add_argument("--patience", default=10, type=int)

    parser.add_argument("--threshold", default=0.0, type=float)

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

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    load_dotenv()

    config = get_config()
    run_path = os.path.join("pt", config.run_name)
    os.makedirs(run_path, exist_ok=True)
    # 경로 생성
    set_experiment_dir(
        config.run_name,
        config.save_dir,
        config.ckpt_dir,
        config.logging_dir,
        config.absolute_dir,
    )
    # huggingface hub, wandb 로그인
    login()
    wandb.login()
    wandb.init(
        project=config.project_name,
        name=config.run_name,
    )
    do_pre_train(config)
    wandb.finish()

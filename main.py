import argparse
import os

# import wandb
from huggingface_hub import login
from dotenv import load_dotenv

from model import do_train, kfold_do_train
from utils import set_experiment_dir


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
        default="test",
        type=str,
        help="wandb에서 쓰일 run_name",
    )

    parser.add_argument(
        "--logging-steps",
        default=10,
        type=int,
        help="wandb에서 업데이트 될 step 기준",
    )

    parser.add_argument(
        "--use-local-zip",
        default=1,
        choices=[0, 1],
        type=int,
    )

    parser.add_argument(
        "--dataset-dir",
        default="100suping/malpyeong-hate-speech",
        type=str,
        help="허깅페이스 허브에 있는 데이터셋 경로 [user_id/repo_name]",
    )

    parser.add_argument(
        "--dataset-revision",
        default="main",
        type=str,
        choices=["main", "delete", "drop"],
        help="허깅페이스 허브에 있는 데이터셋의 브랜치",
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
        default="model",
        type=str,
        help="로컬에 모델을 저장할 때, exp/--run-name 아래에 모델과 config가 저장될 디렉터리",
    )

    parser.add_argument(
        "--ckpt-dir",
        default="ckpts",
        type=str,
        help="로컬에 모델 체크포인트를 저장할 시, exp/--run-name 아래에 체크포인트가 저장될 디렉터리",
    )

    parser.add_argument(
        "--logging-dir",
        default="logs",
        type=str,
        help="로컬에 로깅을 진행 할 시, exp/--run-name 아래에 로깅이 저장될 디렉터리",
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

    parser.add_argument(
        "--neftune-noise-alpha",
        type=int,
        default=0.1,
        help="학습 시 임베딩 벡터에 노이즈를 추가하여 성능을 향상시킬 수 있는 허깅페이스 옵션",
    )

    parser.add_argument("--patience", default=3, type=int)

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
    
    parser.add_argument(
        "--K_Fold",
        type=int,
        default=5,
    )
    
    parser.add_argument(
        "--K_Fold_Train",
        type=bool,
        default=False
    )

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    load_dotenv()

    config = get_config()
    run_path = os.path.join("exp", config.run_name)
    os.makedirs(run_path, exist_ok=True)
    # 경로 생성
    set_experiment_dir(
        config.run_name, config.save_dir, config.ckpt_dir, config.logging_dir
    )
    # huggingface hub, wandb 로그인
    login()
    # wandb.login()
    # wandb.init(
    #     project=config.project_name,
    #     name=config.run_name,
    # )
    if config.K_Fold_Train:
        kfold_do_train(config)
    else:
        do_train(config)
    # wandb.finish()

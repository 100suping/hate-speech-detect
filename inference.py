from model import inference
import argparse


def get_config():

    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Hyperparameters",
    )

    parser.add_argument(
        "--run-name",
        default="test",
        type=str,
        help="train에서 쓰인 run_name",
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
        "--save-dir",
        default="model",
        type=str,
    )

    parser.add_argument(
        "--model-name",
        default="beomi/korean-hatespeech-multilabel",
        choices=["beomi/korean-hatespeech-multilabel", "matthewburke/korean_sentiment"],
        type=str,
    )

    parser.add_argument(
        "--max-len",
        default=20,
        type=int,
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
        "--result-dir",
        type=str,
        default="result",
    )

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    inference_config = get_config()

    inference(inference_config)

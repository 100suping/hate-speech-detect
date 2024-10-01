from model import inference
import argparse


def get_config():

    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Hyperparameters",
    )

    parser.add_argument(
        "--dataset-dir",
        default="/root/exp/NIKL_AU_2023_COMPETITION_v1.0",
        type=str,
    )

    parser.add_argument(
        "--zip-path",
        default="/root/exp/data/NIKL_AU_2023_v1.0_JSONL.zip",
        type=str,
    )

    parser.add_argument(
        "--save-dir",
        default="/root/exp/model",
        type=str,
    )

    parser.add_argument(
        "--model-name",
        default="beomi/korean-hatespeech-multilabel",
        choices=["beomi/korean-hatespeech-multilabel", "matthewburke/korean_sentiment"],
        type=str,
    )

    parser.add_argument(
        "--max_len",
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

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    inference_config = get_config()

    inference(inference_config)

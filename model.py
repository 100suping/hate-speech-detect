from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback,
)
import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader

from data import get_dataset
from utils import MyTrainer, MyTrainerCallback, compute_metrics, set_seed


def load_model_and_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.num_labels == 1:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        if config.model_name == "beomi/korean-hatespeech-multilabel":
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name, num_labels=config.num_labels
            )

    return model, tokenizer


def load_model_and_tokenizer_for_inference(save_dir, model_name, run_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(run_name, save_dir), local_files_only=True
    )

    return model, tokenizer


def make_trainer(train_config, model, tokenizer):
    # Arguments
    training_args = TrainingArguments(
        output_dir=train_config.ckpt_dir,  # output directory
        num_train_epochs=train_config.epochs,  # total number of training epochs
        per_device_train_batch_size=train_config.batch_size,  # batch size per device during training default: 8
        per_device_eval_batch_size=train_config.batch_size,  # batch size for evaluation default: 8
        warmup_steps=train_config.warmup_steps,  # number of warmup steps for learning rate scheduler
        learning_rate=train_config.lr,
        weight_decay=train_config.weight_decay,  # strength of weight decay
        fp16=train_config.fp16,  # device가 CUDA일때만 사용 가능하다.
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,  # n steps 만큼 가중치를 업데이트 하지 않고, 한번에 업데이트
        logging_dir=train_config.logging_dir,  # directory for storing logs
        logging_strategy="epoch",
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        eval_strategy="epoch",  # evalute after each epoch
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        run_name=train_config.run_name,  # [matthewburke/korean_sentiment, beomi/korean-hatespeech-multilabel]
        report_to="wandb",
        # disable_tqdm=True,
        seed=42,  # Seed for experiment reproducibility 3x3
        neftune_noise_alpha=train_config.neftune_noise_alpha,
    )

    # 데이터
    train_dataset, valid_dataset = get_dataset(
        train_config, tokenizer=tokenizer, type_="train", submission=False
    ), get_dataset(train_config, tokenizer=tokenizer, type_="valid", submission=False)

    # optimizer, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    # scheduler 생성 시 고려사항
    # num_training_steps, num_warmup_steps: 총 iteration 수(epoch * iteration) 이다. 따라서 gradient_accumulation_steps또한 고려해야 한다.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=(
            (len(train_dataset) - 1) // training_args.train_batch_size + 1
        )
        * training_args.num_train_epochs,
    )

    early_stopping = EarlyStoppingCallback(
        train_config.patience, train_config.threshold
    )

    loss_name = (
        "CrossEntropy"
        if (train_config.model_name == "beomi/korean-hatespeech-multilabel")
        and (train_config.num_labels == 2)
        else None
    )
    my_trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping, MyTrainerCallback],
        optimizers=(optimizer, lr_scheduler),
        loss_name=loss_name,
    )
    return my_trainer


def do_train(config):
    # seed값 고정
    set_seed(config.seed)

    # device 지정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}.")

    # 모델
    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)

    # trainer 생성
    trainer = make_trainer(config, model, tokenizer)

    # 학습
    trainer.train()

    model.save_pretrained(os.path.join(config.run_name, config.save_dir))


def inference(config):
    os.makedirs(config.result_dir, exist_ok=True)
    # seed값 고정
    set_seed(config.seed)

    # device 지정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}.")

    model, tokenizer = load_model_and_tokenizer_for_inference(
        config.save_dir, config.model_name, config.run_name
    )
    model.to(device)

    test_dataset = get_dataset(
        config, tokenizer=tokenizer, type_="test", submission=False
    )
    submission_df = get_dataset(
        config, tokenizer=tokenizer, type_="test", submission=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    answer = []
    print("Inference Start!")
    for data in tqdm(test_loader):
        input_ids = data["input_ids"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        if outputs.logits.shape[1] == 1:
            predictions = np.where(outputs.logits.detach().cpu().numpy() > 0.5, 1, 0)
        else:
            predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)

        answer.append(predictions)

    answer = np.concatenate(answer, axis=0)

    from datetime import datetime

    now = datetime.now()
    timestamp = f"{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

    submission_df["output"] = answer
    result_path = os.path.join(config.result_dir, timestamp)
    submission_df.to_json(
        f"{result_path}.json",
        orient="records",
        force_ascii=False,
        lines=True,
    )

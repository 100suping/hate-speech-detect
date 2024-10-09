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

from data import (
    get_dataset_hf,
    get_dataset,
    data_collator_for_label_reshape,
    data_collator,
    kfold_datasets
)
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

# KFold 용 load_model_and_tokenizer_forinference, fold 값을 받아 save_dir/model_name/fold 경로 지정
def KFold_load_model_and_tokenizer_for_inference(save_dir, run_name, fold):
    save_path = os.path.join("exp", run_name, save_dir, fold)
    tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        save_path, local_files_only=True
    )
    return model, tokenizer

def load_model_and_tokenizer_for_inference(save_dir, run_name):
    save_path = os.path.join("exp", run_name, save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        save_path, local_files_only=True
    )

    return model, tokenizer


def make_trainer(train_config, model, tokenizer):
    ckpt_path = os.path.join("exp", train_config.ckpt_dir)
    # Arguments
    training_args = TrainingArguments(
        output_dir=ckpt_path,  # output directory
        num_train_epochs=train_config.epochs,  # total number of training epochs
        per_device_train_batch_size=train_config.batch_size,  # batch size per device during training default: 8
        per_device_eval_batch_size=train_config.batch_size,  # batch size for evaluation default: 8
        warmup_steps=train_config.warmup_steps,  # number of warmup steps for learning rate scheduler
        learning_rate=train_config.lr,
        weight_decay=train_config.weight_decay,  # strength of weight decay
        # fp16=train_config.fp16,  # device가 CUDA일때만 사용 가능하다.
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,  # n steps 만큼 가중치를 업데이트 하지 않고, 한번에 업데이트
        neftune_noise_alpha=train_config.neftune_noise_alpha,
        logging_dir=train_config.logging_dir,  # directory for storing logs
        logging_strategy="steps",
        logging_steps=train_config.logging_steps,
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        eval_strategy="epoch",  # evalute after each epoch
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        run_name=train_config.run_name,
        # report_to="wandb",
        # disable_tqdm=True,
        seed=42,  # Seed for experiment reproducibility 3x3
    )

    # 데이터
    if train_config.use_local_zip:
        print("You're using Data from Local")
        train_dataset, valid_dataset = get_dataset(
            train_config, tokenizer=tokenizer, type_="train", submission=False
        ), get_dataset(
            train_config, tokenizer=tokenizer, type_="valid", submission=False
        )

    else:
        print("You're using Data from Huggingface-hub")
        train_dataset, valid_dataset = get_dataset_hf(
            train_config, tokenizer=tokenizer, type_="train", submission=False
        ), get_dataset_hf(
            train_config, tokenizer=tokenizer, type_="valid", submission=False
        )

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
        data_collator=(
            None if train_config.use_local_zip else data_collator_for_label_reshape
        ),
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
    save_path = os.path.join("exp", config.run_name, config.save_dir)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def inference(config):
    result_path = os.path.join("exp", config.result_dir)
    os.makedirs(result_path, exist_ok=True)
    # seed값 고정
    set_seed(config.seed)

    # device 지정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}.")

    model, tokenizer = load_model_and_tokenizer_for_inference(
        config.save_dir, config.run_name
    )
    model.to(device)

    if config.use_local_zip:
        print("You're using Data from Local")
        test_dataset = get_dataset(
            config, tokenizer=tokenizer, type_="test", submission=False
        )
        submission_df = get_dataset(
            config, tokenizer=tokenizer, type_="test", submission=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=32,
            shuffle=False,
        )
    else:
        print("You're using Data from Huggingface-hub")
        test_dataset = get_dataset_hf(
            config, tokenizer=tokenizer, type_="test", submission=False
        )
        submission_df = get_dataset_hf(
            config, tokenizer=tokenizer, type_="test", submission=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator
        )

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
    result_path = os.path.join("exp", config.result_dir, timestamp)
    submission_df.to_json(
        f"{result_path}.json",
        orient="records",
        force_ascii=False,
        lines=True,
    )
    
def ensemble(result_path):
    import json
    from collections import defaultdict, Counter
    import glob
    
    # result 폴더에 이름이 result_ 인 json 파일 모두 불러오기
    model_files = glob.glob(f'{result_path}/result_*.json')

    # 각 모델의 예측 결과를 저장할 리스트
    model_predictions = []

    # 각 모델 파일을 한 줄씩 읽어 리스트에 추가
    for file in model_files:
        with open(file, 'r', encoding='utf-8') as f:
            # 각 모델의 예측을 담을 리스트
            predictions = []
            for line in f:
                # 각 줄을 json 객체로 파싱하여 predictions 리스트에 추가
                predictions.append(json.loads(line.strip()))
            model_predictions.append(predictions)
            # print(file) # 확인용

    # 각 id에 대해 예측값을 저장할 딕셔너리 생성
    votes = defaultdict(list)
    input_texts = {}

    # 각 모델의 예측값을 샘플별로 모으기
    for prediction in model_predictions:
        for sample in prediction:
            sample_id = sample['id']
            sample_input = sample['input']
            sample_output = sample['output']
            votes[sample_id].append(sample_output)
            # 각 샘플의 input 텍스트를 저장 (모든 모델에 동일하므로 한 번만 저장)
            if sample_id not in input_texts:
                input_texts[sample_id] = sample_input

    # Hard Voting 결과 계산
    hard_voting_result = []
    for sample_id, pred_list in votes.items():
        # 각 샘플의 예측값 리스트에서 다수결을 통한 최종 클래스 선택
        final_prediction = Counter(pred_list).most_common(1)[0][0]
        # 최종 결과에 input, id, output 모두 포함
        hard_voting_result.append({
            "id": sample_id,
            "input": input_texts[sample_id],
            "output": final_prediction
        })

    # 최종 결과를 json 파일로 저장
    with open(f"./{result_path}/hard_voting_result.json", 'w', encoding='utf-8') as f:
        for entry in hard_voting_result:
            # 한 줄씩 JSON 객체로 저장하여 모델 출력 파일과 같은 형식으로 만듦
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("결과가 저장되었습니다")

def kfold_do_train(config):
    """kfold 학습"""
    # seed값 고정
    set_seed(config.seed)

    # device 지정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}.")

    # 모델
    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)
    
    # KFold로 나눈 데이터셋
    fold_datasets = kfold_datasets(config, tokenizer, k=config.K_Fold)

    # Fold만큼 반복
    for fold, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f"--------Training fold {fold + 1}--------")
        
        # 두 번째 학습 시작 시 model, tkenizer 재정의, 각 폴드별 학습을 분리하기 위함
        if fold >= 1:
            model, tokenizer = load_model_and_tokenizer(config)
        
        # Trainer 생성
        trainer = make_trainer(config, model, tokenizer)
        
        # 학습
        trainer.train_dataset = train_dataset # 학습 데이터 재정의
        trainer.eval_dataset = val_dataset # 검증 데이터 재정의
        trainer.train()
        
        # 저장
        save_path = save_path = os.path.join("exp", config.run_name, config.save_dir, f"{fold+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
def kfold_inference(config):
    """kfold 추론"""
    result_path = os.path.join("exp", config.result_dir)
    os.makedirs(result_path, exist_ok=True)
    # seed값 고정
    set_seed(config.seed)

    # device 지정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}.")
    
    # 각 1~K번 폴더에 있는 모델 추론
    for fold in range(1,config.K_Fold+1):
        model, tokenizer = KFold_load_model_and_tokenizer_for_inference(
            config.save_dir, config.run_name, f"{fold}"
        )
        model.to(device)

        if config.use_local_zip:
            print("You're using Data from Local")
            test_dataset = get_dataset(
                config, tokenizer=tokenizer, type_="test", submission=False
            )
            submission_df = get_dataset(
                config, tokenizer=tokenizer, type_="test", submission=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=32,
                shuffle=False,
            )
        else:
            print("You're using Data from Huggingface-hub")
            test_dataset = get_dataset_hf(
                config, tokenizer=tokenizer, type_="test", submission=False
            )
            submission_df = get_dataset_hf(
                config, tokenizer=tokenizer, type_="test", submission=True
            )
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator
            )
            
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

        submission_df["output"] = answer
        result_path_fold = os.path.join("exp", config.result_dir, f"result_{fold}")
        submission_df.to_json(
            f"{result_path_fold}.json",
            orient="records",
            force_ascii=False,
            lines=True,
        )
        
    ensemble(result_path)
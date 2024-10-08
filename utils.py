from transformers import Trainer, TrainerCallback
import torch
import numpy as np
import random, os
from sklearn.metrics import f1_score


class MyTrainer(Trainer):
    def __init__(self, loss_name=None, *args, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)
        self.loss_name = loss_name
        print(
            f"Loss funciton {self.loss_name} has set Forcingly."
            if loss_name
            else "No Loss function was set Forcingly."
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        from - https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3522
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            # 추가된 부분 - labels를 모델의 입력으로 주지 않기
            # 특정 모델들은 num_labels=2 이면, 자동으로 이진분류로 인식하여 마지막 레이어의 출력 노드가 2임에도 불구하고,
            # 강제로 BCELoss를 적용시킨다. 따라서, 이를 막기 위해서는 input의 label을 제거하고 모델에 입력해야 한다.
            if self.loss_name == "CrossEntropy":
                labels = inputs.pop("labels")
            else:
                labels = None

        # 추가된 부분 - Loss funciton 강제 지정
        if self.loss_name == "CrossEntropy":
            custom_loss = torch.nn.CrossEntropyLoss()
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # 추가된 부분 - Loss 계산
        # labels is not None 을 시작으로 분기처리 되는 부분의 하위 분기에서 custom loss를 계산할 시,
        # _is_peft_model가 존재하지 않는다는 오류가 발생한다.
        if self.loss_name == "CrossEntropy":
            loss = custom_loss(outputs.logits, labels.squeeze(dim=-1))
        else:
            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()

                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class MyTrainerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training!!")


def compute_metrics(pred):
    labels = pred.label_ids
    if pred.predictions.shape[1] == 1:
        preds = np.where(pred.predictions > 0.5, 1, 0)
    else:
        preds = np.argmax(pred.predictions, axis=1)

    f1 = f1_score(labels, preds, average="macro")

    return {"f1": f1}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed value is {seed}")


def set_experiment_dir(run_name, save_dir, ckpt_dir, logging_dir):
    """argparse로 받은 경로들을 이용하여 실험의 각 요소들이 저장될 디렉터리를 만들어주는 함수입니다."""
    save_path = os.path.join("exp", run_name, save_dir)
    ckpt_path = os.path.join("exp", run_name, ckpt_dir)
    logging_path = os.path.join("exp", run_name, logging_dir)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(logging_path, exist_ok=True)


def is_bool(value):
    if value == "none":
        return None
    if value == "true" or value == "True":
        return True
    else:
        False


def is_int_float(value):
    if value == "none":
        return None
    if isinstance(value, (int, float, None)):
        return value
    else:
        raise TypeError


def is_int(value):
    if value == "none":
        return None
    if isinstance(value, (int, None)):
        return value
    else:
        raise TypeError


def is_float(value):
    if value == "none":
        return None
    if isinstance(value, (float, None)):
        return value
    else:
        raise TypeError

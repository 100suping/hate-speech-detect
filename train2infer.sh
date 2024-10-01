python3 main.py \
  --zip-path /root/exp/data/NIKL_AU_2023_v1.0_JSONL.zip \
  --dataset-dir /root/exp/NIKL_AU_2023_COMPETITION_v1.0 \
  --model-type electra \
  --model-name beomi/korean-hatespeech-multilabel \
  --save-dir /root/exp/model\
  --ckpt-dir /root/exp/ckpts\
  --num-labels 1 \
  --epochs 3 \
  --batch-size 32 \
  --max-len 20 \
  --warmup-steps 2 \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --fp16 True \
  --gradient-accumulation-steps 10 \
  --patience 3 \
  --run-name test \
  --test-run 1 \

python3 inference.py \
  --dataset-dir /root/exp/NIKL_AU_2023_COMPETITION_v1.0 \
  --zip-path /root/exp/data/NIKL_AU_2023_v1.0_JSONL.zip \
  --save-dir /root/exp/model \
  --model-name beomi/korean-hatespeech-multilabel \
  --max-len 20 \
  --test-run 1
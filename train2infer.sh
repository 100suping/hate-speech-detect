python3 main.py \
  --dataset-dir 100suping/malpyeong-hate-speech \
  --dataset-revision main \
  --model-type electra \
  --model-name beomi/korean-hatespeech-multilabel \
  --run-name test \
  --save-dir model\
  --ckpt-dir ckpts\
  --num-labels 1 \
  --epochs 1 \
  --batch-size 32 \
  --max-len 20 \
  --warmup-steps 25 \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --fp16 True \
  --gradient-accumulation-steps 2 \
  --neftune-noise-alpha 5 \
  --patience 3 \
  --test-run 1 \

python3 inference.py \
  --dataset-dir 100suping/malpyeong-hate-speech \
  --dataset-revision main \
  --save-dir model \
  --model-name beomi/korean-hatespeech-multilabel \
  --max-len 20 \
  --test-run 1
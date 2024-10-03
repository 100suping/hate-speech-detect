python /home/hiyo2044/hate-speech-detect/main.py \
    --zip-path /home/hiyo2044/hate-speech-detect/data/NIKL_AU_2023_v1.0_JSONL.zip \
    --dataset-dir /home/hiyo2044/hate-speech-detect/NIKL_AU_2023_COMPETITION_v1.0 \
    --save-dir /home/hiyo2044/hate-speech-detect/model \
    --ckpt-dir /home/hiyo2044/hate-speech-detect/ckpts \
    --logging-dir /home/hiyo2044/hate-speech-detect/logs \
    --neftune_noise_alpha 5

python /home/hiyo2044/hate-speech-detect/inference.py \
    --dataset-dir /home/hiyo2044/hate-speech-detect/NIKL_AU_2023_COMPETITION_v1.0 \
    --zip-path /home/hiyo2044/hate-speech-detect/data/NIKL_AU_2023_v1.0_JSONL.zip \
    --save-dir /home/hiyo2044/hate-speech-detect/model \
    --result_dir /home/hiyo2044/hate-speech-detect/result 

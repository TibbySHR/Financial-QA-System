CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path /udata/dingmin/LLaMA-Factory/models/Qwen1.5-7B/ \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --export_dir merge_qwen1.5-7b-intent-v2/ \
    --adapter_name_or_path ./saves/ckpt-7b/intent-v2/checkpoint-660/

# 训练方式——原生

CUDA_VISIBLE_DEVICES=0 python3 finetune.py # device_map="auto"指定无效

# 训练方式——LLaMA-Factory

1. git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
2. 将训练数据放置于LLaMA-Factory/data目录下
3. 在data/dataset_info.json中注册训练数据
4. CUDA_VISIBLE_DEVICES=0 llamafactory-cli train config/llama3_lora_dpo.yaml
5. llamafactory-cli export config/llama3_merge_lora.yaml
# -*- coding: utf-8 -*

import os
# from modelscope import snapshot_download
# from transformers.training_args import OptimizerNames
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd
from transformers.training_args import OptimizerNames
from prompt_helper import *

# GPU最佳设置
torch.cuda.empty_cache()  # 清理缓存区
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 模型下载
model_dir = "/opt/models/Meta-Llama-3-8B-Instruct"
lora_model_dir = "/opt/models/Comm100-Llama-3-8B-Instruct-Lora"

# 模型加载
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
print(model)  # 打印模型及参数架构
model.config.use_cache = False
model.enable_input_require_grads()

# tokenizer加载
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
print(tokenizer.pad_token, tokenizer.eos_token)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token_id, tokenizer.eos_token_id, "model loaded successfully.")


# 数据处理
def process_func(example):
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['Instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False)
    response = tokenizer(f"{example['Answer']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > 4500:
        print("data too long.----------------{}".format(example['Answer']))
        return {"input_ids": [], "attention_mask": [], "labels": []}
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


df = pd.read_json('data/train.json')
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
max_tokens_in_train = sorted([len(tokenizer(" ".join(df.iloc[row_idx].astype(str)), add_special_tokens=False)["input_ids"]) for row_idx in range(len(df))], reverse=True)[:10]
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id = tokenized_id.filter(lambda example_: bool(example_['input_ids']))
print("data processed successfully, length: {}, max_tokens_in_train: {}".format(len(tokenized_id), max_tokens_in_train))

print("start training...")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 模型训练
args = TrainingArguments(
    output_dir=lora_model_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=1,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=5,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    bf16=True,
    tf32=True,
    optim=OptimizerNames.PAGED_ADAMW
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
trainer.train()
# trainer.train(resume_from_checkpoint=True)  # 断点续训
print("model trained successfully.")

# 模型保存
trainer.model.save_pretrained(lora_model_dir)
print("LoRA model saved successfully.")

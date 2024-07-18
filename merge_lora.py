# -*- coding: utf-8 -*

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

start_time = time.time()

model_dir = "/opt/share/chatglm/models/meta/Meta-Llama-3-8B-Instruct"
lora_model_dir = "/opt/qs/models/Comm100-Llama-3-8B-Instruct-Lora"
finetuned_model_dir = "/opt/qs/models/Comm100-Llama-3-8B-Instruct"

# merge model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_model_dir)
model = model.merge_and_unload()  # CPU上执行merge操作，合并时选用的是最后一次保存的adapter

model.save_pretrained(finetuned_model_dir)
tokenizer.save_pretrained(finetuned_model_dir)
print("model merged successfully, cost: {} seconds.".format(time.time() - start_time))

# inference测试
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir, use_fast=False, trust_remote_code=True)
sys_message = "You are a trustworthy AI assistant."
messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": "你是谁？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

model = AutoModelForCausalLM.from_pretrained(finetuned_model_dir, device_map="auto", torch_dtype=torch.bfloat16)
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.encode('<|eot_id|>')[0]
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

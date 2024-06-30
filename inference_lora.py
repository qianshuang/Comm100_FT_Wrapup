# -*- coding: utf-8 -*

import os
import time
import traceback

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from utils import *
from prompt_helper import *

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

start_time = time.time()

model_dir = "/opt/models/Meta-Llama-3-70B-Instruct"
lora_model_dir = "/opt/models/Comm100-Llama-3-70B-Instruct-Lora"

# merge model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 不需要添加
model = AutoPeftModelForCausalLM.from_pretrained(lora_model_dir, device_map="auto", torch_dtype=torch.bfloat16)
print("model loaded successfully, cost: {} seconds.".format(time.time() - start_time))

csv_res_file = "data/test_result_Comm100.csv"
test_data = load_json_file("data/test.json")
instructions = []
pred_answers = []
answers = []
for i, data in enumerate(test_data):
    start_time = time.time()
    print("{} is inferencing, length: {}...".format(i + 1, len(data["Instruction"].split())))

    match = re.search(r'# CHAT HISTORY #\n(.*?)\n# RETURN AS A JSON #', data["Instruction"], re.DOTALL)
    dialogue = match.group(1).strip()
    instructions.append(dialogue)

    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": data["Instruction"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    try:
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.encode('<|eot_id|>')[0])
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
    except:
        traceback.print_exc()
        response = ""
    pred_answers.append(response)
    answers.append(data["Answer"])

    cost = time.time() - start_time
    print("{} finished, cost: {} seconds.".format(i + 1, cost))

df = pd.DataFrame({'instruction': instructions, 'pred_answer': pred_answers, 'answer': answers})
df.to_csv(csv_res_file, index=False)
print("test result inference and saved successfully.")

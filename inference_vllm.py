# -*- coding: utf-8 -*

import time

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import *
from prompt_helper import *

model_path = "/opt/qs/models/Comm100-Llama-3-8B-Instruct"
llm = LLM(model=model_path, tensor_parallel_size=1, dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0, max_tokens=4096, stop_token_ids=[tokenizer.encode('<|eot_id|>')[0]])

csv_res_file = "data/test_result.csv"
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
    output = llm.generate([text], sampling_params)

    pred_answer = output[0].outputs[0].text
    print(pred_answer)

    pred_answers.append(pred_answer)
    answers.append(data["Answer"])

    cost = time.time() - start_time
    print("{} finished, cost: {} seconds.".format(i + 1, cost))

df = pd.DataFrame({'instruction': instructions, 'pred_answer': pred_answers, 'answer': answers})
df.to_csv(csv_res_file, index=False)
print("test result inference and saved successfully.")

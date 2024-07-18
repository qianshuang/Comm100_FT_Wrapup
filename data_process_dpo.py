# -*- coding: utf-8 -*

import random

import pandas as pd
from utils import *
from prompt_helper import *

# 数据预处理，拆分训练测试集
# columns_to_convert_2022 = ['Content', 'Category', 'Wrapup.Case Status']
# columns_to_convert_2024 = ['Content', 'Category', 'Case Status']
columns_to_convert = ['Content', 'Category', 'Case_Status']
#
# df_2022 = pd.read_excel("data/Chats_data_042022.xlsx", engine='openpyxl')
# df_2024 = pd.read_excel("data/Chats_data_052024.xlsx", engine='openpyxl')
#
# df_2022 = df_2022[columns_to_convert_2022]
# df_2024 = df_2024[columns_to_convert_2024]
#
# df_2022.columns = columns_to_convert
# df_2024.columns = columns_to_convert

df_gaming_0628 = pd.read_excel("data/chats_20240628_gaming_finetune.xlsx", engine='openpyxl')
df_0708 = pd.read_csv("data/Chats_20240708.csv", encoding='utf-8')
df_gaming_0628 = df_gaming_0628[["Content", "Category"]]
df_gaming_0628['Case_Status'] = ''
df_0708 = df_0708[["Content", "Category"]]
df_0708['Case_Status'] = ''

df_test = df_0708.head(200)
df_train_remain = df_0708.drop(df_test.index)
df_test.to_csv('data/test.csv', encoding='utf-8', index=False)

# df_combined = pd.concat([df_2022, df_2024, df_gaming_0628, df_train_remain], ignore_index=True)
df_combined = pd.concat([df_gaming_0628, df_train_remain], ignore_index=True)
df_combined.drop_duplicates(inplace=True)
df_combined[columns_to_convert] = df_combined[columns_to_convert].astype(str)
print(df_combined)

uniq_category = df_combined["Category"].unique()
uniq_cs = df_combined["Case_Status"].unique()
uniq_category = uniq_category[uniq_category != 'nan']
uniq_cs = uniq_cs[uniq_cs != 'nan']
print("uniq_category: {}, uniq_cs: {}".format(uniq_category, uniq_cs))

df_combined.to_csv('data/train.csv', encoding='utf-8', index=False)
print("train length: {}, test length: {}".format(len(df_combined), len(df_test)))


# 生成finetune data.json
def process_test_data(in_file, out_file):
    qa_dict_list = []
    df_ = pd.read_csv(in_file, encoding='utf-8')
    for index, row in df_.iterrows():
        if str(row['Content']).strip() in ["", "nan"]:
            continue
        qa_dict = {"Instruction": instruction_template.format(row['Content'].strip()),
                   "Answer": json.dumps(
                       {
                           "Category": "" if str(row['Category']).strip() == "nan" else row['Category'].strip(),
                           "Case_Status": "" if str(row['Case_Status']).strip() == "nan" else row['Case_Status'].strip()
                       }
                   )}
        qa_dict_list.append(qa_dict)
    write_json_file(qa_dict_list, out_file)
    print("qa_dict length：{}".format(len(qa_dict_list)))


def process_train_data(in_file, out_file):
    qa_dict_list = []
    df_ = pd.read_csv(in_file, encoding='utf-8')
    for index, row in df_.iterrows():
        if str(row['Content']).strip() in ["", "nan"]:
            continue
        qa_dict = {"conversations": [{"from": "system", "value": sys_message}, {"from": "human", "value": instruction_template.format(row['Content'].strip())}],
                   "chosen": {"from": "gpt", "value": json.dumps({
                       "Category": "" if str(row['Category']).strip() == "nan" else row['Category'].strip(),
                       "Case_Status": "" if str(row['Case_Status']).strip() == "nan" else row['Case_Status'].strip()})},
                   "rejected": {"from": "gpt", "value": json.dumps({
                       "Category": random.choice(uniq_category),
                       "Case_Status": random.choice(uniq_cs)})}}
        qa_dict_list.append(qa_dict)
    write_json_file(qa_dict_list, out_file)
    print("qa_dict length：{}".format(len(qa_dict_list)))


process_train_data("data/train.csv", "data/Comm100_dpo_train.json")
process_test_data("data/test.csv", "data/test.json")

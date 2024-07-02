# -*- coding: utf-8 -*

import pandas as pd
from utils import *
from prompt_helper import *

# 数据预处理，拆分训练测试集
# columns_to_convert_2022 = ['Content', 'Category', 'Wrapup.Case Status', 'Wrapup.Profit Center']
# columns_to_convert_2024 = ['Content', 'Category', 'Case Status', 'Profit Center']
# columns_to_convert = ['Content', 'Category', 'Case_Status', 'Profit_Center']
#
# df_2022 = pd.read_excel("data/Chats_data_042022.xlsx", engine='openpyxl')
# df_2024 = pd.read_excel("data/Chats_data_052024.xlsx", engine='openpyxl')
#
# df_2022 = df_2022[columns_to_convert_2022]
# df_2024 = df_2024[columns_to_convert_2024]
#
# df_2022.columns = columns_to_convert
# df_2024.columns = columns_to_convert
#
# df_combined = pd.concat([df_2022, df_2024], ignore_index=True)
# df_combined.drop_duplicates(inplace=True)
# df_combined[columns_to_convert] = df_combined[columns_to_convert].astype(str)
# print(df_combined)
#
# test_set = df_combined.sample(n=200, random_state=42)
# train_set = df_combined.drop(test_set.index)
#
# test_set.to_csv('data/test.csv', index=False)
# train_set.to_csv('data/train.csv', index=False)
# print("train length: {}, test length: {}".format(len(train_set), len(test_set)))

# 补充训练数据
df_pre = pd.read_csv('data/train.csv', encoding='utf-8')
df = pd.read_excel("data/chats_20240628_gaming_finetune.xlsx", engine='openpyxl')
df = df[["Content", "Category"]]
df['Case_Status'] = ''
df['Profit_Center'] = ''

df_combined = pd.concat([df_pre, df], ignore_index=True)
df_combined.drop_duplicates(inplace=True)
df_combined.to_csv('data/train.csv', encoding='utf-8', index=False)
print("train length: {}".format(len(df_combined)))


# 生成finetune data.json
def process_data(in_file, out_file):
    qa_dict_list = []
    df_ = pd.read_csv(in_file, encoding='utf-8')
    for index, row in df_.iterrows():
        if str(row['Content']).strip() in ["", "nan"]:
            continue
        qa_dict = {"Instruction": instruction_template.format(row['Content'].strip()),
                   "Answer": json.dumps(
                       {
                           "Category": "" if str(row['Category']).strip() == "nan" else row['Category'].strip(),
                           "Case_Status": "" if str(row['Case_Status']).strip() == "nan" else row['Case_Status'].strip(),
                           "Profit_Center": "" if str(row['Profit_Center']).strip() == "nan" else row['Profit_Center'].strip()
                       }
                   )}
        qa_dict_list.append(qa_dict)
    write_json_file(qa_dict_list, out_file)
    print("qa_dict length：{}".format(len(qa_dict_list)))


process_data("data/train.csv", "data/train.json")
# process_data("data/test.csv", "data/test.json")

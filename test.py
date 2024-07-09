# -*- coding: utf-8 -*

import pandas as pd
from utils import *
from prompt_helper import *


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


# 数据预处理，拆分训练测试集
columns_to_convert = ['Content', 'Category']
df = pd.read_csv("data/Chats_20240708.csv", encoding='utf-8')
df = df[columns_to_convert]
df.drop_duplicates(inplace=True)
df[columns_to_convert] = df[columns_to_convert].astype(str)
df['Case_Status'] = ''
df['Profit_Center'] = ''
print(df)

test_set = df.head(200)
test_set.to_csv('data/test_gaming.csv', index=False)
print(test_set)

process_data("data/test_gaming.csv", "data/test_gaming.json")

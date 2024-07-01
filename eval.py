# -*- coding: utf-8 -*

import pandas as pd
import json

# 读取 CSV 文件
df = pd.read_csv("data/test_result.csv")


def calculate_score(row):
    answer = json.loads(row['answer'])
    pred_answer = json.loads(row['pred_answer'])

    score = 0
    for key in answer:
        if key in pred_answer and answer[key] == pred_answer[key]:
            score += 1
    return score


df['score'] = df.apply(calculate_score, axis=1)
print(df.head())
# df.to_csv('data/test_result_with_score.csv', index=False)
print("Score: {}".format(df['A'].sum() / (3 * len(df))))

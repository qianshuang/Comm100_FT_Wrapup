# -*- coding: utf-8 -*

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# from sklearn.svm import SVC

# train_df = pd.read_csv("data/train.csv", encoding='utf-8')
# test_df = pd.read_csv("data/test.csv", encoding='utf-8')
# train_df = train_df[["Content", "Category"]]
# test_df = test_df[["Content", "Category"]]
# train_df = train_df.dropna()
# test_df = test_df.dropna()

df = pd.read_excel("data/train_1_GPT4o筛选.xlsx", engine='openpyxl')
df = df[["Content", "Category"]]
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# train_df = pd.read_excel("data/train_1_GPT4o筛选.xlsx", engine='openpyxl')
# train_df = train_df[["Content", "Category"]]
# test_df = pd.read_csv("data/test.csv", encoding='utf-8')
# test_df = test_df[["Content", "Category"]]
# print(len(test_df))

# 1. 特征提取：将文本数据转换为数值特征
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['Content'])
X_test = vectorizer.transform(test_df['Content'])
print("finishing vectorized...")

# 2. 标签
y_train = train_df['Category']
y_test = test_df['Category']

# 3. 训练随机森林分类器
clf = RandomForestClassifier()
# clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
print("finishing training...")

# 4. 预测并计算准确率
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集上的准确率: {accuracy:.2f}")

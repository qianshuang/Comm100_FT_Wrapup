# -*- coding: utf-8 -*

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("data/train.csv", encoding='utf-8')
test_df = pd.read_csv("data/test.csv", encoding='utf-8')
train_df = train_df[["Content", "Category"]]
test_df = test_df[["Content", "Category"]]
train_df = train_df.dropna()
test_df = test_df.dropna()

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
clf.fit(X_train, y_train)
print("finishing training...")

# 4. 预测并计算准确率
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集上的准确率: {accuracy:.2f}")

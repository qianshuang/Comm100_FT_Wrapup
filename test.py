# -*- coding: utf-8 -*

import pandas as pd

# 创建一个示例 DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 27, 22],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)

# 将 DataFrame 转换为 Markdown 格式的字符串
markdown_table = df.to_markdown(index=False)
print(markdown_table)

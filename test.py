# -*- coding: utf-8 -*

import pandas as pd
import numpy as np

# 示例DataFrame
data = {
    'A': [1, 2, 2, 3, 4, 4, 4, np.nan],
    'B': ['a', 'b', 'b', 'c', 'd', 'd', 'd', 'e']
}
df = pd.DataFrame(data)

# 一行代码获取列 'A' 的所有唯一值并去掉 NaN
unique_values = df['A'].dropna().unique()

print(unique_values)

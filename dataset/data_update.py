import pandas as pd
import numpy as np
np.random.seed(5900)
remove_n = 3000
#train = pd.read_csv("tokened_traindataset.csv")
#train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]

#df = pd.DataFrame(train)

#print(df.groupby("sentiment").count())

#df2 = df[df['sentiment'] == 5]

#df2.to_csv("num5.csv", mode='w')

remove5 = pd.read_csv("remove_5.csv")
only5 = pd.read_csv("num5_final.csv")

df1 = pd.DataFrame(remove5)
df2 = pd.DataFrame(only5)

data = df1.append(df2)

print(data)
print(data.groupby("sentiment").count())

"""
count = 2900
while count > 0:
    df2.drop([np.random.randint(1,5909)])
    count = count-1
"""

data.to_csv("remove5_dataset.csv", mode='w', index=False)



import pandas as pd
import numpy as np

origin_train = pd.read_csv("add_sad_train_final1.csv")
origin_test = pd.read_csv("add_sad_test1.csv")

df = pd.DataFrame(origin_train)
df2 = pd.DataFrame(origin_test)

only_joy_train = df[df['sentiment'] == 1]
only_sad_train = df[df['sentiment'] == 2]
only_neutral_train = df[df['sentiment'] == 5]
only_upset_train = df[df['sentiment'] == 3]

only_joy_test = df2[df2['sentiment'] == 1]
only_sad_test = df2[df2['sentiment'] == 2]
only_neutral_test = df2[df2['sentiment'] == 5]
only_upset_test = df2[df2['sentiment'] == 3]

only_joy_train.to_csv("only_joy_train.csv",mode='w', index=False)
only_sad_train.to_csv("only_sad_train.csv",mode='w', index=False)
only_neutral_train.to_csv("only_neutral_train.csv",mode='w', index=False)
only_upset_train.to_csv("only_upset_train.csv",mode='w', index=False)

only_joy_test.to_csv("only_joy_test.csv",mode='w', index=False)
only_sad_test.to_csv("only_sad_test.csv",mode='w', index=False)
only_neutral_test.to_csv("only_neutral_test.csv",mode='w', index=False)
only_upset_test.to_csv("only_upset_test.csv",mode='w', index=False)


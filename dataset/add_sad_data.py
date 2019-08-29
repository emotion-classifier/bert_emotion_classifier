import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

add_sad = pd.read_csv("sad_train.csv")
origin_train_data = pd.read_csv("remove5_dataset.csv")
origin_test_data = pd.read_csv("tokened_testdataset.csv")

#데이터셋 분리
sad_train, sad_test = train_test_split(add_sad, test_size=0.20, random_state=42)

#분리된 데이터셋 확인
sad_train["sentiment"] = pd.Categorical(sad_train["sentiment"])

print(sad_train.groupby("sentiment").count())
print(sad_test.groupby("sentiment").count())

new_train_df = pd.DataFrame(origin_train_data)
new_test_df = pd.DataFrame(origin_test_data)

sad_train_df = pd.DataFrame(sad_train)
sad_test_df = pd.DataFrame(sad_test)

final_train = new_train_df.append(sad_train_df)
final_test = new_test_df.append(sad_test_df)

print(final_train.groupby("sentiment").count())
print(final_test.groupby("sentiment").count())

final_train.to_csv("add_sad_train1.csv", mode='w', index=False)
final_test.to_csv("add_sad_test1.csv", mode='w', index=False)
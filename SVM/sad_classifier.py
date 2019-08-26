import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

train = pd.read_csv("sad_train_data.csv")
test = pd.read_csv("sad_test_data.csv")
sad_text = pd.read_csv("only_0_text.csv")

train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
sad_text = sad_text[~sad_text.isin([np.nan, np.inf, -np.inf]).any(1)]

#데이터셋 분리
#train, test = train_test_split(data, test_size=0.20, random_state=42)

#분리된 데이터셋 확인
train["sentiment"] = pd.Categorical(train["sentiment"])

print(train.groupby("sentiment").count())
print(test.groupby("sentiment").count())

"""
le = LabelEncoder()
le.fit(train['tweet'].astype(str))
train['tweet'] = le.transform(train['tweet'].astype(str))
"""

#명사 추출하여 TDM 생성
def get_noun(text):
    tokenizer = Okt()
    nouns = tokenizer.nouns(text)
    return [n for n in nouns]

cv = CountVectorizer(tokenizer=get_noun)
tdm = cv.fit_transform(train["tweet"].values.astype('U'))

#print(cv.vocabulary_)

#SVM으로 분류학습

text_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=get_noun)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', OneClassSVM(degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001,
                   nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None))])

#text_clf_svm = OneClassSVM(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001,
#                   nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None)

#le = LabelEncoder()
#le.fit(train['tweet'])
text_clf_svm = text_clf_svm.fit(train["tweet"].values.astype('U'), train["sentiment"].values.astype('U'))

#testSet으로 정확도 평가
predicted_svm = text_clf_svm.predict(test["tweet"])
#print(np.mean(predicted_svm))

#print(predicted_svm[0])
#print(predicted_svm[1])
#print(predicted_svm[2])
#print("len predicted svm :" + str(len(predicted_svm)))

#print("len test data : " + str(len(sad_text)))

    #predicted_svm = text_clf_svm.predict([s_data])
    #if predicted_svm == 1 :
    #print(sad_text["reviews"][0])
    #print(sad_text["reviews"][1])
    #print(sad_text["reviews"][2])

sad_dataFrame = pd.DataFrame(columns=['tweet'])
count = 0


for s_data in sad_text["reviews"]:
    predicted_svm = text_clf_svm.predict([s_data])
    if predicted_svm == 1:
        sad_dataFrame.loc[count] = s_data
        print(s_data)
        print(count)
        #print(sad_dataFrame["tweet"][count])
        count = count + 1
        sad_dataFrame.to_csv('sad_movieData.csv')


"""
predicted_svm = text_clf_svm.predict(sad_text["reviews"])
#print(len(predicted_svm))

for i in predicted_svm:
    if i == 1:
        count = count+1
print(count)
"""

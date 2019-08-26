import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
"""Bag of words + SVM"""

#data = pd.read_csv("utf8_complete_tweetlist.csv")

#빈값이 들어있는 것 제거(sklearn 오류 때문)
#data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

train = pd.read_csv("add_sad_train_final1.csv")
test = pd.read_csv("add_sad_test1.csv")

train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]

#데이터셋 분리
#train, test = train_test_split(data, test_size=0.20, random_state=42)

#분리된 데이터셋 확인
train["sentiment"] = pd.Categorical(train["sentiment"])

print(train.groupby("sentiment").count())
print(test.groupby("sentiment").count())

#명사 추출하여 TDM 생성
def get_noun(text):
    tokenizer = Okt()
    nouns = tokenizer.nouns(text)
    return [n for n in nouns]

#Bag of words
cv = CountVectorizer(tokenizer=get_noun)
tdm = cv.fit_transform(train["tweet"].values.astype('U'))

#print(cv.vocabulary_)

#SVM으로 분류학습
text_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=get_noun)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(train["tweet"].values.astype('U'), train["sentiment"])

#testSet으로 정확도 평가
predicted_svm = text_clf_svm.predict(test["tweet"])
print(predicted_svm)
print(np.mean(predicted_svm == test["sentiment"]))

print(text_clf_svm.predict(["진짜 예쁘다", "열애 보도를 인정했다", "큰 부상이 아니었으면", "씨발"]))


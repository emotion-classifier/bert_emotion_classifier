import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
"""Bag of words + SVM"""
"""binary classifier"""

#read csv
joy_test = pd.read_csv("add_sad_train_final1.csv")
joy_train = pd.read_csv("only_joy_train.csv")

neutral_test = pd.read_csv("add_sad_train_final1.csv")
neutral_train = pd.read_csv("only_neutral_train.csv")

sad_test = pd.read_csv("add_sad_train_final1.csv")
sad_train = pd.read_csv("only_sad_train.csv")

upset_test = pd.read_csv("add_sad_train_final1.csv")
upset_train = pd.read_csv("only_upset_train.csv")

#mapping
#1
joy_test['sentiment'] = joy_test['sentiment'].map({1: 1, 2: -1, 3: -1, 5: -1})
joy_train['sentiment'] = joy_train['sentiment'].map({1: 1})

#5
neutral_test['sentiment'] = neutral_test['sentiment'].map({1: -1, 2: -1, 3: -1, 5: 1})
neutral_train['sentiment'] = neutral_train['sentiment'].map({5: 1})

#2
sad_test['sentiment'] = sad_test['sentiment'].map({1: -1, 2: 1, 3: -1, 5: -1})
sad_train['sentiment'] = sad_train['sentiment'].map({2: 1})

#3
upset_test['sentiment'] = upset_test['sentiment'].map({1: -1, 2: -1, 3: 1, 5: -1})
upset_train['sentiment'] = upset_train['sentiment'].map({3: 1})


#데이터셋 분리
#train, test = train_test_split(data, test_size=0.20, random_state=42)

#분리된 데이터셋 확인
upset_train["sentiment"] = pd.Categorical(upset_train["sentiment"])

print("CHECK DATASET")
print("JOY>>>>>>>>>>")
print(joy_train.groupby("sentiment").count())
print(joy_test.groupby("sentiment").count())
print("\n")
print("NEUTRAL>>>>>>")
print(neutral_train.groupby("sentiment").count())
print(neutral_test.groupby("sentiment").count())
print("\n")
print("SAD>>>>>>>>>>")
print(sad_train.groupby("sentiment").count())
print(sad_test.groupby("sentiment").count())
print("\n")
print("UPSET>>>>>>>>")
print(upset_train.groupby("sentiment").count())
print(upset_test.groupby("sentiment").count())
print("\n")

#명사 추출하여 TDM 생성
def get_noun(text):
    tokenizer = Okt()
    nouns = tokenizer.nouns(text)
    return [n for n in nouns]

#Bag of words
cv = CountVectorizer(tokenizer=get_noun)

tdm = cv.fit_transform(joy_train["tweet"].values.astype('U'))
tdm = cv.fit_transform(neutral_train["tweet"].values.astype('U'))
tdm = cv.fit_transform(sad_train["tweet"].values.astype('U'))
tdm = cv.fit_transform(upset_train["tweet"].values.astype('U'))


#SVM으로 분류학습
joy_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=get_noun)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', OneClassSVM(kernel="rbf", degree=3, gamma="scale",
                         coef0=0.0, tol=1e-3, nu=0.1, shrinking=True, cache_size=100, verbose=False,
                         max_iter=-1, random_state=42))])

neutral_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=get_noun)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', OneClassSVM(kernel="rbf", degree=3, gamma="scale",
                         coef0=0.0, tol=1e-3, nu=0.1, shrinking=True, cache_size=100, verbose=False,
                         max_iter=-1, random_state=42))])

sad_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=get_noun)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', OneClassSVM(kernel="rbf", degree=3, gamma="scale",
                         coef0=0.0, tol=1e-3, nu=0.1, shrinking=True, cache_size=100, verbose=False,
                         max_iter=-1, random_state=42))])

upset_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=get_noun)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', OneClassSVM(kernel="rbf", degree=3, gamma="scale",
                         coef0=0.0, tol=1e-3, nu=0.1, shrinking=True, cache_size=100, verbose=False,
                         max_iter=-1, random_state=42))])


joy_clf_svm = joy_clf_svm.fit(joy_train["tweet"].values.astype('U'), joy_train["sentiment"])
neutral_clf_svm = neutral_clf_svm.fit(neutral_train["tweet"].values.astype('U'), neutral_train["sentiment"])
sad_clf_svm = sad_clf_svm.fit(sad_train["tweet"].values.astype('U'), sad_train["sentiment"])
upset_clf_svm = upset_clf_svm.fit(upset_train["tweet"].values.astype('U'), upset_train["sentiment"])



count = 0
#testSet으로 정확도 평가
print("\n")
print("JOY ACCURACY")
joy_predicted_svm = joy_clf_svm.predict(joy_test["tweet"])
joy_accuracy = np.mean(joy_predicted_svm == joy_test["sentiment"])
print(joy_predicted_svm)
print(joy_accuracy)
joy_flag = True
while(joy_flag == True):
    for tweet in joy_test["tweet"]:
        predicted_svm = joy_clf_svm.predict([tweet])
        if count > 100:
            count = 0
            joy_flag = False
            break
        if predicted_svm == 1:
            print(tweet)
            count = count+1
print("\n")


print("NEUTRAL ACCURACY")
neutral_predicted_svm = neutral_clf_svm.predict(neutral_test["tweet"])
neutral_accuracy = np.mean(neutral_predicted_svm == neutral_test["sentiment"])
print(neutral_predicted_svm)
print(neutral_accuracy)
neutral_flag = True
while(neutral_flag == True):
    for tweet in neutral_test["tweet"]:
        predicted_svm = neutral_clf_svm.predict([tweet])
        if count > 100:
            count = 0
            neutral_flag = False
            break
        if predicted_svm == 1:
            print(tweet)
            count = count+1
print("\n")


print("SAD ACCURACY")
sad_predicted_svm = sad_clf_svm.predict(sad_test["tweet"])
sad_accuracy = np.mean(sad_predicted_svm == sad_test["sentiment"])
print(sad_predicted_svm)
print(sad_accuracy)
sad_flag = True
while(sad_flag == True):
    for tweet in sad_test["tweet"]:
        predicted_svm = sad_clf_svm.predict([tweet])
        if count > 100:
            count = 0
            sad_flag = False
            break
        if predicted_svm == 1:
            print(tweet)
            count = count+1
print("\n")


print("UPSET ACCURACY")
upset_predicted_svm = upset_clf_svm.predict(upset_test["tweet"])
upset_accuracy = np.mean(upset_predicted_svm == upset_test["sentiment"])
print(upset_predicted_svm)
print(upset_accuracy)
upset_flag = True
while(upset_flag == True):
    for tweet in upset_test["tweet"]:
        predicted_svm = upset_clf_svm.predict([tweet])
        if count > 100:
            count = 0
            upset_flag = False
            break
        if predicted_svm == 1:
            print(tweet)
            count = count+1
print("\n")

print("CLASSIFIER USING TEXT")
test_list1 = [" 넹 ㅋㅋ 영 드 리 메이크 작 이 고 정경호 주연이 이 ᆫ데 저 는 재밌 게 보 었 어요 !",
                           "이지오 선수 의료진 과 함께 휠체어 로 퇴장 ㅠㅠ",
                           "이진숙 대전 MBC 사장 , 해임 앞두 고 자진 사퇴 - 뻔뻔 하 기 ᆫ ! 퇴직금 은 무신 , 사퇴 받 어 주 지 말 고 해임 하 기 ᆯ 바라 ᆫ다 . ",
                           "한비야 결혼 사실 뒤늦 게 화제 … 남편 은 네덜란드 출신 긴급 구호 전문가 "]
test_list2 = ["진짜 예쁘다", "열애 보도를 인정했다", "애도", "놈"]
test_list3 = ["방탄", "보도했다", "슬픔", "자한당"]

print("joy", joy_clf_svm.predict(test_list1))
print("neutral", neutral_clf_svm.predict(test_list1))
print("sad", sad_clf_svm.predict(test_list1))
print("upset", upset_clf_svm.predict(test_list1))
print("\n")
print("joy", joy_clf_svm.predict(test_list2))
print("neutral", neutral_clf_svm.predict(test_list2))
print("sad", sad_clf_svm.predict(test_list2))
print("upset", upset_clf_svm.predict(test_list2))
print("\n")
print("joy", joy_clf_svm.predict(test_list3))
print("neutral", neutral_clf_svm.predict(test_list3))
print("sad", sad_clf_svm.predict(test_list3))
print("upset", upset_clf_svm.predict(test_list3))


print("\n")
print("JOY ACCURACY : ", joy_accuracy)
print("NEUTRAL ACCURACY : ", neutral_accuracy)
print("SAD ACCURACY : ", sad_accuracy)
print("UPSET ACCURACY : ", upset_accuracy)

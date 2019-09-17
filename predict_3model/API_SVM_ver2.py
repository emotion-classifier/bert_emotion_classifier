import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
""" SVM API """

def get_noun(text):
    tokenizer = Okt()
    nouns = tokenizer.nouns(text)
    return [n for n in nouns]

def support_vector_api(tweet):
    print("로딩중입니다")
    with open('support_vector.model', 'rb') as f:
        text_clf_svm = pickle.load(f)
        #cv = pickle.load(f)

    #tfidf_transformer = TfidfTransformer()
    #Xtest2 = cv.transform([tweet])
    #tfidfv_t2 = tfidf_transformer.fit_transform(Xtest2)

    return text_clf_svm.predict([tweet])[0]

if __name__ == "__main__":
    print("트윗을 입력하세요 : ")
    tweet = input()
    val_predict = support_vector_api(tweet)
    if val_predict == 1:
        print("기쁨")
    elif val_predict == 2:
        print("슬픔")
    elif val_predict == 3:
        print("화남")
    elif val_predict == 5:
        print("중립")
    else:
        print(val_predict[0], "잘못된 분류값입니다.")
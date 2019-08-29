import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from sklearn.svm import OneClassSVM
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

test = pd.read_csv("add_sad_train_final1.csv")
train = pd.read_csv("sad_train.csv")

test['sentiment'] = test['sentiment'].map({1: -1, 2: 1, 3: -1, 5: -1})
train['sentiment'] = train['sentiment'].map({2: 1})

#print(test.shape)
#print(train.shape)
#train['sentiment'] = train['sentiment'].map({'2': 1})
#test['sentiment'] = test['sentiment'].map({'1.0': -1, '2.0': 1, '3.0': -1, '5.0': -1})

#train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
#test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]

train_tweet = train['tweet'].tolist()
train_sentiment = train['sentiment'].tolist()

test_tweet = test['tweet'].tolist()
test_sentiment = test['sentiment'].tolist()

#명사 추출하여 TDM 생성
def get_noun(text):
    tokenizer = Okt()
    nouns = tokenizer.nouns(text)
    return [n for n in nouns]

print(get_noun(train_tweet[9]))

vectorizer = HashingVectorizer(n_features=20, tokenizer=get_noun)
features = vectorizer.fit_transform(train_tweet).toarray()

print(features.shape)

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
pipe_clf = Pipeline([('vectorizer', vectorizer), ('clf', clf)])
pipe_clf.fit(train_tweet, train_sentiment)

preds_train = pipe_clf.predict(train_tweet)
print("Accuracy:", accuracy_score(train_sentiment, preds_train))
print(np.mean(preds_train == 1))
print(np.mean(preds_train))

preds_test = pipe_clf.predict(test_tweet)
results = confusion_matrix(test_sentiment, preds_test)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(test_sentiment, preds_test))
print('Report : ')
print(classification_report(test_sentiment, preds_test))
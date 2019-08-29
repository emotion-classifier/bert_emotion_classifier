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


#check dataset
print(joy_test.shape)
print(joy_train.shape)

print(neutral_test.shape)
print(neutral_train.shape)

print(sad_test.shape)
print(sad_train.shape)

print(upset_test.shape)
print(upset_train.shape)


#transform to list
joy_train_tweet = joy_train['tweet'].tolist()
joy_train_sentiment = joy_train['sentiment'].tolist()
joy_test_tweet = joy_test['tweet'].tolist()
joy_test_sentiment = joy_test['sentiment'].tolist()

neutral_train_tweet = neutral_train['tweet'].tolist()
neutral_train_sentiment = neutral_train['sentiment'].tolist()
neutral_test_tweet = neutral_test['tweet'].tolist()
neutral_test_sentiment = neutral_test['sentiment'].tolist()

sad_train_tweet = sad_train['tweet'].tolist()
sad_train_sentiment = sad_train['sentiment'].tolist()
sad_test_tweet = sad_test['tweet'].tolist()
sad_test_sentiment = sad_test['sentiment'].tolist()

upset_train_tweet = upset_train['tweet'].tolist()
upset_train_sentiment = upset_train['sentiment'].tolist()
upset_test_tweet = upset_test['tweet'].tolist()
upset_test_sentiment = upset_test['sentiment'].tolist()


#명사 추출하여 TDM 생성
def get_noun(text):
    tokenizer = Okt()
    nouns = tokenizer.nouns(text)
    return [n for n in nouns]

print(get_noun(sad_train_tweet[9]))

vectorizer = HashingVectorizer(n_features=20, tokenizer=get_noun)

joy_features = vectorizer.fit_transform(joy_train_tweet).toarray()
neutral_features = vectorizer.fit_transform(neutral_train_tweet).toarray()
sad_features = vectorizer.fit_transform(sad_train_tweet).toarray()
upset_features = vectorizer.fit_transform(upset_train_tweet).toarray()

print(joy_features.shape)
print(neutral_features.shape)
print(sad_features.shape)
print(upset_features.shape)

#SVM model
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

joy_pipe_clf = Pipeline([('vectorizer', vectorizer), ('clf', clf)])
neutral_pipe_clf = Pipeline([('vectorizer', vectorizer), ('clf', clf)])
sad_pipe_clf = Pipeline([('vectorizer', vectorizer), ('clf', clf)])
upset_pipe_clf = Pipeline([('vectorizer', vectorizer), ('clf', clf)])

joy_pipe_clf.fit(joy_train_tweet, joy_train_sentiment)
neutral_pipe_clf.fit(neutral_train_tweet, neutral_train_sentiment)
sad_pipe_clf.fit(sad_train_tweet, sad_train_sentiment)
upset_pipe_clf.fit(upset_train_tweet, upset_train_sentiment)

#evaluation
#joy
print("JOY EVALUATION\n")

print("TEST SET ACCURACY")
preds_joy_train = joy_pipe_clf.predict(joy_train_tweet)
print("Accuracy:", accuracy_score(joy_train_sentiment, preds_joy_train) , "\n")

print("TRAIN SET ACCURACY")
preds_joy_test = joy_pipe_clf.predict(joy_test_tweet)
results = confusion_matrix(joy_test_sentiment, preds_joy_test)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(joy_test_sentiment, preds_joy_test))
print('Report : ')
print(classification_report(joy_test_sentiment, preds_joy_test), "\n")


#neutral
print("NEUTRAL EVALUATION\n")

print("TEST SET ACCURACY")
preds_neutral_train = neutral_pipe_clf.predict(neutral_train_tweet)
print("Accuracy:", accuracy_score(neutral_train_sentiment, preds_neutral_train), "\n")

print("TRAIN SET ACCURACY")
preds_neutral_test = neutral_pipe_clf.predict(neutral_test_tweet)
results = confusion_matrix(neutral_test_sentiment, preds_neutral_test)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(neutral_test_sentiment, preds_neutral_test))
print('Report : ')
print(classification_report(neutral_test_sentiment, preds_neutral_test), "\n")


#sad
print("SAD EVALUATION\n")

print("TEST SET ACCURACY")
preds_sad_train = sad_pipe_clf.predict(sad_train_tweet)
print("Accuracy:", accuracy_score(sad_train_sentiment, preds_sad_train), "\n")

print("TRAIN SET ACCURACY")
preds_sad_test = sad_pipe_clf.predict(sad_test_tweet)
results = confusion_matrix(sad_test_sentiment, preds_sad_test)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(sad_test_sentiment, preds_sad_test))
print('Report : ')
print(classification_report(sad_test_sentiment, preds_sad_test), "\n")


#upset
print("UPSET EVALUATION\n")

print("TEST SET ACCURACY")
preds_upset_train = upset_pipe_clf.predict(upset_train_tweet)
print("Accuracy:", accuracy_score(upset_train_sentiment, preds_upset_train), "\n")

print("TRAIN SET ACCURACY")
preds_upset_test = upset_pipe_clf.predict(upset_test_tweet)
results = confusion_matrix(upset_test_sentiment, preds_upset_test)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(upset_test_sentiment, preds_upset_test))
print('Report : ')
print(classification_report(upset_test_sentiment, preds_upset_test), "\n")


#evaluation
print("EVALUATION (USING TEXT)")

print("\n")
print(joy_test_tweet[1])
print(neutral_test_tweet[11803])
print(sad_test_tweet[7])
print(upset_test_tweet[747])
print("\n")
print(joy_test_sentiment[1])
print(neutral_test_sentiment[11803])
print(sad_test_sentiment[7])
print(upset_test_sentiment[747])
print("\n")
print(joy_pipe_clf.predict([joy_test_tweet[1]]))
print(neutral_pipe_clf.predict([neutral_test_tweet[11803]]))
print(sad_pipe_clf.predict([sad_test_tweet[7]]))
print(upset_pipe_clf.predict([upset_test_tweet[747]]))


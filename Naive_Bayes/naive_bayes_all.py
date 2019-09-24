from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import numpy as np

files = ['', 'half_neutral_', 'add_sad_',
            'add_sad_half_neutral_', 'tokened_',
            'tokened_half_neutral_', 'tokened_add_sad_',
            'tokened_add_sad_half_neutral_']


for filename in files:
    #train dataset 로딩

    newsdata = {'data' : [], 'target' : []}

    f = open(filename + 'train_data.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        try:
            newsdata['target'].append(line[1])
            newsdata['data'].append(line[0])
        except:
            pass
        
    f.close()

    newsdata['target'] = np.array(newsdata['target'])



    #test dataset 로딩

    newsdata_test = {'data' : [], 'target' : []}

    f = open(filename + 'test_data.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        try:
            newsdata_test['target'].append(line[1])
            newsdata_test['data'].append(line[0])
        except:
            pass

    f.close()

    newsdata_test['target'] = np.array(newsdata_test['target'])



    # 나이브 베이즈 분류

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(newsdata['data'])

    tfidf_transformer = TfidfTransformer()
    tfidfv = tfidf_transformer.fit_transform(X_train_dtm)

    mod = MultinomialNB()
    mod.fit(tfidfv, newsdata['target'])

    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    X_test_dtm = dtmvector.transform(newsdata_test['data']) 
    tfidfv_test = tfidf_transformer.transform(X_test_dtm)

    predicted = mod.predict(tfidfv_test)
    print(filename + " 정확도:", accuracy_score(newsdata_test['target'], predicted))

input()

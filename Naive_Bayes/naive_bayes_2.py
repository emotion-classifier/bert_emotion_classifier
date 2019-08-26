from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import numpy as np


#train dataset 로딩

datacount = [0,0,0,0,0,0]
flag = 0

newsdata = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

newsdata_test = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

f = open('tweetlist(ansi).csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    try:
        if(1<=int(line[1])<=5):
            if(int(line[1]) != 5):
                newsdata['data'].append(line[0])
                newsdata['target'].append(int(line[1]))
                datacount[int(line[1])] = datacount[int(line[1])] + 1
                flag = flag + 1
            elif(flag != 0):
                newsdata['data'].append(line[0])
                newsdata['target'].append(int(line[1]))
                datacount[int(line[1])] = datacount[int(line[1])] + 1
                flag = flag -1 
            
    except:
        pass
    
f.close()

newsdata_test['target'] = list(newsdata['target'])[1600:]
newsdata_test['data'] = list(newsdata['data'])[1600:]

newsdata['target'] = newsdata['target'][:1600]
newsdata['data'] = newsdata['data'][:1600]

newsdata['target'] = np.array(newsdata['target'])
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
print("정확도:", accuracy_score(newsdata_test['target'], predicted))

mod.predict()

input()

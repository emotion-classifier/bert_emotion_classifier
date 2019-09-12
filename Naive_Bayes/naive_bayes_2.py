from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import numpy as np


#train dataset 로딩

datacount = [0,0,0,0,0,0]
datacount_test = [0,0,0,0,0,0]
flag = 0

newsdata = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

newsdata_test = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

f = open('final_traindata.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    try:
        if(1<=int(line[1])<=5):
            if(int(line[1]) != 5):
                newsdata['data'].append(line[0])
                newsdata['target'].append(int(line[1]))
                datacount[int(line[1])] = datacount[int(line[1])] + 1
                flag = flag + 1
            elif(flag == 4):
                newsdata['data'].append(line[0])
                newsdata['target'].append(int(line[1]))
                datacount[int(line[1])] = datacount[int(line[1])] + 1
                flag = flag - 4 
            
    except:
        pass
    
f.close()

f = open('final_testdata.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    try:
        if(1<=int(float(line[1]))<=5):
            if(int(float(line[1])) != 5):
                newsdata_test['data'].append(line[0])
                newsdata_test['target'].append(int(float(line[1])))
                datacount_test[int(line[1])] = datacount_test[int(float(line[1]))] + 1
                flag = flag + 1
            elif(flag == 4):
                newsdata_test['data'].append(line[0])
                newsdata_test['target'].append(int(line[1]))
                datacount_test[int(line[1])] = datacount_test[int(float(line[1]))] + 1
                flag = flag - 4 
            
    except:
        pass
    
f.close()


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


input()

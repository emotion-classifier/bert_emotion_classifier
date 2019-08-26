from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import numpy as np


#train dataset 로딩

newsdata = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

f = open('add_sad_train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    try:
        if(1<=int(line[1])<=5):
            newsdata['data'].append(line[0])
            newsdata['target'].append(int(line[1]))
    except:
        pass
    
##    if(line[1]=='기쁨' or line[1]=='슬픔' or line[1]=='화남' or line[1]=='중립'):
##        newsdata['data'].append(line[0])
##    
##    if(line[1]=='기쁨'):
##        newsdata['target'].append(1)
##    if(line[1]=='슬픔'):
##        newsdata['target'].append(2)
##    if(line[1]=='화남'):
##        newsdata['target'].append(3)
##    if(line[1]=='중립'):
##        newsdata['target'].append(5)
    
f.close()

newsdata['target'] = np.array(newsdata['target'])



#test dataset 로딩

newsdata_test = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

f = open('add_sad_test.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    try:
        if(1<=int(line[1])<=5):
            newsdata_test['data'].append(line[0])
            newsdata_test['target'].append(int(line[1]))
    except:
        pass
##    if(line[1]=='기쁨' or line[1]=='슬픔' or line[1]=='화남' or line[1]=='중립'):
##        newsdata_test['data'].append(line[0])
    
##    if(line[1]=='기쁨'):
##        newsdata_test['target'].append(1)
##    if(line[1]=='슬픔'):
##        newsdata_test['target'].append(2)
##    if(line[1]=='화남'):
##        newsdata_test['target'].append(3)
##    if(line[1]=='중립'):
##        newsdata_test['target'].append(5)
    
f.close()

newsdata_test['target'] = np.array(newsdata_test['target'])


print(len(newsdata['target']))
print(len(newsdata_test['target']))

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

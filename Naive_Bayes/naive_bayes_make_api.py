from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import pickle


#train dataset 로딩

datacount = [0,0,0,0,0,0]
datacount_test = [0,0,0,0,0,0]
flag = 0

newsdata = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

newsdata_test = {'data' : [], 'target' : [], 'target_names' :
            ['기쁨', '슬픔', '화남', '불안', '중립']}

f = open('tokened_balanced_add_sad_half_neutral_train_data.csv', 'r', encoding='utf-8')
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


# 나이브 베이즈 분류

dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(newsdata['data'])

tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)

mod = MultinomialNB()
mod.fit(tfidfv, newsdata['target'])

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# 모델 파일 생성

with open('naive_bayes.model', 'wb') as f:
    pickle.dump(mod, f)
    pickle.dump(dtmvector, f)


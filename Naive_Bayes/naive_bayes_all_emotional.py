from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import numpy as np

files = ['', 'tokened_', 'half_neutral_', 'tokened_half_neutral_',             
            'add_sad_','tokened_add_sad_', 'add_sad_half_neutral_', 
            'tokened_add_sad_half_neutral_']


for filename in files:
    #train dataset 로딩

    newsdata = {'data' : [], 'target' : []}

    f = open(filename + 'train_data.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        try:
            newsdata['target'].append(line[1].strip())
            newsdata['data'].append(line[0])
        except:
            pass
        
    f.close()

    newsdata['target'] = np.array(newsdata['target'])



    #test dataset 로딩

    test_target = [[],[],[],[]]
    emotions = [[],[],[],[]]

    f = open(filename + 'test_data.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        try:
            data = int(float(line[1]))
            if(1 <= data <= 3):
                test_target[data-1].append(line[0])
                emotions[data-1].append(line[1].strip())
            else:
                test_target[3].append(line[0])
                emotions[3].append(line[1].strip())
        except:
            pass

    f.close()

    for i in range(4):
        emotions[i] = np.array(emotions[i])


    # 나이브 베이즈 분류

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(newsdata['data'])

    tfidf_transformer = TfidfTransformer()
    tfidfv = tfidf_transformer.fit_transform(X_train_dtm)

    mod = MultinomialNB()
    mod.fit(tfidfv, newsdata['target'])

    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    print(filename + " 정확도: \t")
    
    for i in range(4):
        X_test_dtm = dtmvector.transform(test_target[i]) 
        tfidfv_test = tfidf_transformer.transform(X_test_dtm)

        predicted = mod.predict(tfidfv_test)
        cnt = 0
        for j in range(len(predicted)):
            
            if(int(predicted[j]) == int(emotions[i][j])):
                cnt = cnt + 1
        print(str(float(cnt / len(predicted))) + " ")
        #print(accuracy_score(emotions[i], predicted) + " ")

input()

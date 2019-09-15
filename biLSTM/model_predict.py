from gensim.models.wrappers import FastText
from keras.models import load_model
import numpy as np

print("load model...")
model_ft = FastText.load_fasttext_format("model_drama.bin")
model = load_model("model/rec-final_traindata.csv-29-0.7666.hdf5")
def featurize_rnn(corpus,wdim,maxlen):
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            pass#print(i)
        s = corpus[i]
        for j in range(len(s)):
            if s[-j-1] in model_ft and j < maxlen:
                rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
    return rnn_total

while True:
    text = [input("텍스트를 입력하세요 : ")]
    print(text)
    text_rnn = featurize_rnn(text,100,120)
    val_predict = np.argmax(np.asarray(model.predict(text_rnn)), axis=1)
    if val_predict[0] == 1:
        print("기쁨")
    elif val_predict[0] == 2:
        print("슬픔")
    elif val_predict[0] == 3:
        print("화남")
    elif val_predict[0] == 5:
        print("중립")
    else:
        print(val_predict[0], "잘못된 분류값입니다.")


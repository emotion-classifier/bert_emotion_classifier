from gensim.models.wrappers import FastText
from keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')


print("load model...")
model_ft = FastText.load_fasttext_format("model_drama.bin")
model = load_model("model/rec-balanced_add_sad_train_data.csv-30-0.7204.hdf5")
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

def call_api(text):
    text_rnn = featurize_rnn(text,100,200)
    val_predict = np.argmax(np.asarray(model.predict(text_rnn)), axis=1)
    ret = ""
    if val_predict[0] == 1:
        ret = "기쁨"
    elif val_predict[0] == 2:
        ret = "슬픔"
    elif val_predict[0] == 3:
        ret = "화남"
    elif val_predict[0] == 5:
        ret = "중립"
    else:
        ret = str(val_predict[0]) + "잘못된 분류값입니다."
    return ret


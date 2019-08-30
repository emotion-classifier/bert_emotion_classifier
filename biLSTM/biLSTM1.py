import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

def read_data(filename):
    import csv
    with open(filename, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        data = list()
        for line in rdr:
            if len(line) != 2:
                print(line)
            else:
                data.append((line[0], int(line[1])))
        #data = [(line[0], line[1]) for line in rdr]
    return data
    
fci = read_data('tokened_traindataset2.csv')
fci_sp_token_train = [t[0] for t in fci]
fci_label_train = [t[1] for t in fci]
fci_test = read_data('tokened_testdataset.csv')
fci_sp_token_test = [t[0] for t in fci_test]
fci_label_test = [t[1] for t in fci_test]

fci_data = fci_sp_token_train + fci_sp_token_test
fci_label = fci_label_train + fci_label_test

import numpy as np
from gensim.models.wrappers import FastText

def featurize_rnn(corpus,wdim,maxlen):
    model_ft = FastText.load_fasttext_format("model_drama.bin")
    print("fasttext model load finished")
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if s[-j-1] in model_ft and j < maxlen:
                rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
    return rnn_total

fci_rec = featurize_rnn(fci_sp_token_train,100,60)#+fci_sp_token_test
predict_test = featurize_rnn(fci_sp_token_test,100,60)#+fci_sp_token_test

from keras.models import Sequential
import keras.layers as layers
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)

class Metricsf1macro(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))#fci_sp_token_test
        val_predict = np.argmax(val_predict,axis=1)

        val_targ = self.validation_data[1]#fci_label_test
        _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
        _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
        _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
        _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
        _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
        _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_f1s_w.append(_val_f1_w)
        self.val_recalls_w.append(_val_recall_w)
        self.val_precisions_w.append(_val_precision_w)
        print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
        print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro = Metricsf1macro()

from sklearn.utils import class_weight
class_weights_fci = class_weight.compute_class_weight('balanced', np.unique(fci_label_train), fci_label_train)

def validate_bilstm(result, y, hidden_lstm, hidden_dim, cw, filename):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_lstm), input_shape=(len(result[0]), len(result[0][0]))))
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro, checkpoint]#
    model.fit(result,y,epochs=30,validation_data=(predict_test, fci_label_test),batch_size=128,callbacks=callbacks_list,class_weight=cw)
    #print(metrics.f1_score(np.argmax(np.asarray(model.predict(predict_test)), axis=1),fci_label_test, average="macro"))

validate_bilstm(fci_rec, fci_label_train, 32, 128, class_weights_fci, 'model/rec')
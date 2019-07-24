import LSTM_Model

model = LSTM_Model.LSTM_Model()

#model.training(epo=10000, continue_training=False, l2_norm=True, first_training=True)

model.keep_prob = 1.0
model.evaluation()

#encoding=utf8
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from PointerLSTM import PointerLSTM
import pickle
import tsp_data as tsp
import numpy as np
from tensorflow import keras
import tensorflow as tf


def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rates
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1
if __name__=='__main__':
        print("preparing dataset...")
        t = tsp.Tsp()
        X, Y = t.next_batch(3)#10000
        #X.shape
        #(3, 10, 2)  
        #Y.shape
        #(3, 10)        
        x_test, y_test = t.next_batch(2)

        YY = []
        for y in Y:
            YY.append(to_categorical(y))
        YY = np.asarray(YY)

        hidden_size = 128
        seq_len = 10
        nb_epochs = 100
        learning_rate = 0.1

        print("building model...")
        main_input = Input(shape=(seq_len, 2), name='main_input')

        #main_input=tf.cast(X,tf.float32)#(3, 10, 2)

        encoder,state_h, state_c = LSTM(hidden_size,return_sequences = True, name="encoder",return_state=True)(main_input)
        decoder = PointerLSTM(hidden_size, name="decoder")(encoder,states=[state_h, state_c])
        print(decoder.shape)
        
        

        model = Model(main_input, decoder)


        ##################用keras方式训练

        print(model.summary())
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X, YY, epochs=nb_epochs, batch_size=64,)
        print('x_test.dtype',x_test.dtype)
        print(model.predict(tf.cast(x_test,tf.float32)))
        print('evaluate : ',model.evaluate(x_test,to_categorical(y_test)))
        print("------")
        print(to_categorical(y_test))
        model.save_weights('model_weight_100.hdf5')

        ###################用tensorflow方式训练

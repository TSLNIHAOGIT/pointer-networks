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
import time


def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rates
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()

# @tf.function
def train_step(inp,targ):
  with tf.GradientTape() as tape:
    encoder = LSTM(hidden_size, return_sequences=True, name="encoder", return_state=True)
    'inp.shape:TensorShape([64, 10, 2]) encoder_output.shape TensorShape([64, 10, 128])'
    encoder_output,state_h, state_c=encoder(inp)
    '''
    (encoder_output[:,-1,:]==state_h).numpy().all() is True
    
    encoder_output返回的所有时序结果中，最后一个时序就是state_h
    '''

    decoder = PointerLSTM(hidden_size, name="decoder")
    '''state_h.shape,           state_c.shape
      (TensorShape([64, 128]), TensorShape([64, 128]))'''
    ##h是输出的状态，c相当于memory
    decoder_output=decoder(encoder_output, training=True,states=[state_h, state_c])
    loss=loss_object(decoder_output,targ)
    print('each batch loss={}'.format(loss.numpy()))

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
if __name__=='__main__':
        print("preparing dataset...")
        t = tsp.Tsp()
        X, Y = t.next_batch(100)#10000
        #X.shape
        #(100, 10, 2)  
        #Y.shape
        #(100, 10)        
        x_test, y_test = t.next_batch(2)

        YY = []
        for y in Y:
            YY.append(to_categorical(y))
        YY = np.asarray(YY)

        hidden_size = 128
        seq_len = 10
        nb_epochs = 100
        learning_rate = 0.1

        # print("building model...")
        # main_input = Input(shape=(seq_len, 2), name='main_input')
        # ##返回返回的是一个list,[encoder,state_h, state_c],其中encoder.shape=(batch_size,time_step,dim),state_h.shape=state_c.shape=(batch_size,dim)
        # encoder,state_h, state_c = LSTM(hidden_size,return_sequences = True, name="encoder",return_state=True)(main_input)
        # decoder = PointerLSTM(hidden_size, name="decoder")(encoder,states=[state_h, state_c])
        #
        # model = Model(main_input, decoder)
        #
        #
        # ##################用keras方式训练
        #
        # print(model.summary())
        # model.compile(optimizer='adam',
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        #
        # model.fit(X, YY, epochs=nb_epochs, batch_size=64,)
        # print('x_test.dtype',x_test.dtype)
        # print(model.predict(x_test))
        # print('evaluate : ',model.evaluate(x_test,to_categorical(y_test)))
        # print("------")
        # print(to_categorical(y_test))
        # model.save_weights('model_weight_100.hdf5')

        ###################用tensorflow方式训练
        X=tf.cast(X,tf.float32)
        # YY=tf.cast(YY,tf.int32)

        EPOCHS = 100
        BUFFER_SIZE = len(X)
        BATCH_SIZE = 64
        steps_per_epoch = len(X) // BATCH_SIZE
        dataset = tf.data.Dataset.from_tensor_slices((X, YY)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


        for epoch in range(EPOCHS):
            start = time.time()


            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                # inp.shape(batch, 10, 2) < dtype: 'float32' > targ.shape(batch, 10, 10) < dtype: 'float32' >
                # print('inp.shape',inp.shape,inp.dtype,'targ.shape',targ.shape,targ.dtype)
                batch_loss = train_step(inp, targ)
                total_loss += batch_loss

                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # # saving (checkpoint) the model every 2 epochs
            # if (epoch + 1) % 2 == 0:
            #     checkpoint.save(file_prefix=checkpoint_prefix)
            #
            # print('Epoch {} Loss {:.4f}'.format(epoch + 1,
            #                                     total_loss / steps_per_epoch))
            # print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


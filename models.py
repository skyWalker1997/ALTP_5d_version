import keras as keras
import numpy as np
import tensorflow as tf
import attention
from keras.layers import Conv2D,Input,LSTM,Reshape,Concatenate,Dense,Activation,Flatten
from keras.models import Model
from keras import backend as K




def model():
    #day1
    input_1 = [Input(shape=(3, 3, 5), name='input_day_1_{0}'.format(i)) for i in range(0, 5)]
    # pre_day_input = [Input(shape=(3, 3, 5),name = 'input_day_{0}_{1}'.format(i,j)) for i in range(0,4)]
    # feature shape = (2,3,3,5)(feature,geo,geo,time)
    conv_1 = [Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 3, 5), name='Conv_day_1_{0}'.format(i))(input_1[i])
              for i
              in range(0, 5)]
    # single out_put_shape = (1,1,1,5)(batch,geo,time)
    # Conv_1 out_put = (4,1,1,1,5)(feature)
    conv_1 = [Reshape((5, 1))(conv_1[i]) for i in range(0, 5)]
    # single out_put_shape = (1,5,1)
    # reshape out_put_shape = (4,1,5,1)(feature,batch,time,feature)
    conv1 = Concatenate(axis=-1)([conv_1[i] for i in range(0, 5)])
    # After concatenate = (1,5,4)(batch,time,feature)
    lstm1 = LSTM(5, input_shape=(5, 5), return_sequences=True, name='att_lstm_day_1')(conv1)
    # After LSTM shape = (1,5,4)(batch,time,feature)

    # day2
    input_2 = [Input(shape=(3, 3, 5), name='input_day_2_{0}'.format(i)) for i in range(0, 5)]
    # feature shape = (4,3,3,5)(feature,geo,geo,time)
    conv_2 = [Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 3, 5), name='Conv_day_2_{0}'.format(i))(input_2[i])
              for i
              in range(0, 5)]
    # single out_put_shape = (1,1,1,5)(batch,geo,time)
    # Conv_1 out_put = (4,1,1,1,5)(feature)
    conv_2 = [Reshape((5, 1))(conv_2[i]) for i in range(0, 5)]
    # single out_put_shape = (1,5,1)
    # reshape out_put_shape = (4,1,5,1)(feature,batch,time,feature)
    conv2 = Concatenate(axis=-1)([conv_2[i] for i in range(0, 5)])
    # After concatenate = (1,5,4)(batch,time,feature)
    lstm2 = LSTM(5, input_shape=(5, 5), return_sequences=True, name='att_lstm_day_2')(conv2)
    # After LSTM shape = (1,5,4)(batch,time,feature)

    # day3
    input_3 = [Input(shape=(3, 3, 5), name='input_day_3_{0}'.format(i)) for i in range(0, 5)]
    # feature shape = (4,3,3,5)(feature,geo,geo,time)
    conv_3 = [Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 3, 5), name='Conv_day_3_{0}'.format(i))(input_3[i])
              for i
              in range(0, 5)]
    # single out_put_shape = (1,1,1,5)(batch,geo,time)
    # Conv_1 out_put = (4,1,1,1,5)(feature)
    conv_3 = [Reshape((5, 1))(conv_3[i]) for i in range(0, 5)]
    # single out_put_shape = (1,5,1)
    # reshape out_put_shape = (4,1,5,1)(feature,batch,time,feature)
    conv3 = Concatenate(axis=-1)([conv_3[i] for i in range(0, 5)])
    # After concatenate = (1,5,4)(batch,time,feature)
    lstm3 = LSTM(5, input_shape=(5, 5), return_sequences=False, name='att_lstm_day_3')(conv3)


    #Knn Input
    # knn_tensor_input = Input(shape=(1,1),name='knn')


    # After LSTM shape = (1,5,4)(batch,time,feature)

    att_lstm1 = attention.Attention(method='cba',name='Attention1')([lstm1, lstm3])
    att_lstm1 = Reshape(target_shape=(1, 5))(att_lstm1)
    att_lstm2 = attention.Attention(method='cba',name='Attention2')([lstm2, lstm3])
    att_lstm2 = Reshape(target_shape=(1, 5))(att_lstm2)
    att_lstm = Concatenate(axis=1)([att_lstm1, att_lstm2])

    # lstm3 = Reshape(target_shape=(1, 4))(lstm3)
    att_high_level = LSTM(5, input_shape=(2, 5), return_sequences=False, name='att_lstm')(att_lstm)

    # att_high_level  = Reshape(target_shape=(1,4))(att_high_level)
    all_lstm = Concatenate(axis=1)([att_high_level, lstm3])
    Dense1_output = Dense(units=10, name='Dense_1')(all_lstm)
    # Dense1_output = Reshape(target_shape=(1,10))(Dense1_output)
    # Dense2 = Concatenate(axis=-1)([Dense1_output, knn_tensor_input])
    # print(Dense2)
    pred = Dense(units=1, name='Dense_2')(Dense1_output)

    # input = Concatenate(axis=1)(input_1+input_2+input_3)

    input = input_1 + input_2 + input_3 #+ [knn_tensor_input]
    model = keras.Model(inputs=input, outputs=pred)
    model.compile(optimizer='adagrad', loss='mae')
    return model
#
# out = model(final_data)
# print(out)

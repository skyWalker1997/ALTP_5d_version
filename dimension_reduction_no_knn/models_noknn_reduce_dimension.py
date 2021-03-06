from dimension_reduction_knn import attention
from keras.layers import Conv2D,Input,LSTM,Reshape,Concatenate,Dense,Add
from keras.models import Model


def model():
    #day1
    input_1 = [Input(shape=(3, 3, 5), name='input_day_1_{0}'.format(i)) for i in range(0, 5)]
    conv_1 = [Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 3, 5), name='Conv_day_1_{0}'.format(i))(input_1[i])
              for i in range(0, 5)]
    conv_1[1] = Concatenate(axis=-1)([conv_1[0],conv_1[1]])
    conv_1[1] = Dense(units=5, name='Dense_cnn_1')(conv_1[1])
    conv_1 = [Reshape((5, 1))(conv_1[i]) for i in range(1, 5)]

    conv_1 = Concatenate(axis=-1)([conv_1[i] for i in range(0, 4)])
    # After concatenate = (1,5,4)(batch,time,feature)
    lstm1 = LSTM(5, input_shape=(5, 4), return_sequences=True, name='att_lstm_day_1')(conv_1)
    # After ALTP_5d_version shape = (1,5,4)(batch,time,feature)

    # day2
    input_2 = [Input(shape=(3, 3, 5), name='input_day_2_{0}'.format(i)) for i in range(0, 5)]
    conv_2 = [Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 3, 5), name='Conv_day_2_{0}'.format(i))(input_2[i])
              for i in range(0, 5)]
    conv_2[1] = Concatenate(axis=-1)([conv_2[0], conv_2[1]])
    conv_2[1] = Dense(units=5, name='Dense_cnn_2')(conv_2[1])
    conv_2 = [Reshape((5, 1))(conv_2[i]) for i in range(1, 5)]
    conv_2 = Concatenate(axis=-1)([conv_2[i] for i in range(0, 4)])
    # After concatenate = (1,5,4)(batch,time,feature)
    lstm2 = LSTM(5, input_shape=(5, 4), return_sequences=True, name='att_lstm_day_2')(conv_2)
    # After ALTP_5d_version shape = (1,5,4)(batch,time,feature)

    # day3
    input_3 = [Input(shape=(3, 3, 5), name='input_day_3_{0}'.format(i)) for i in range(0, 5)]
    # feature shape = (4,3,3,5)(feature,geo,geo,time)
    conv_3 = [Conv2D(filters=5, kernel_size=(3, 3), input_shape=(3, 3, 5), name='Conv_day_3_{0}'.format(i))(input_3[i])
              for i in range(0, 5)]
    conv_3[1] = Concatenate(axis=-1)([conv_3[0], conv_3[1]])
    conv_3[1] = Dense(units=5, name='Dense_cnn_3')(conv_3[1])
    conv_3 = [Reshape((5, 1))(conv_3[i]) for i in range(1, 5)]
    conv3 = Concatenate(axis=-1)([conv_3[i] for i in range(0, 4)])
    # After concatenate = (1,5,4)(batch,time,feature)
    lstm3 = LSTM(5, input_shape=(5, 4), return_sequences=False, name='att_lstm_day_3')(conv3)


    #Knn Input
    # knn_tensor_input = Input(shape=(1,5),name='knn')
    # knn_tensor = Reshape(target_shape=(5,))(knn_tensor_input)


    #*****************
    # knn_tensor = LSTM(1,input_shape=(1,5),return_sequences=False,name='lstm_knn')(knn_tensor_input)
    #*****************



    # After ALTP_5d_version shape = (1,5,4)(batch,time,feature)
    #########################
    att_lstm1 = attention.Attention(method='cba', name='Attention1')([lstm1, lstm3])
    att_lstm1 = Reshape(target_shape=(1, 5))(att_lstm1)
    att_lstm2 = attention.Attention(method='cba', name='Attention2')([lstm2, lstm3])
    att_lstm2 = Reshape(target_shape=(1, 5))(att_lstm2)
    att_lstm = Concatenate(axis=1)([att_lstm1, att_lstm2])
    #################################


    # lstm3 = Reshape(target_shape=(1, 4))(lstm3)

    ###################################
    att_high_level = LSTM(5, input_shape=(2, 5), return_sequences=False, name='att_lstm')(att_lstm)
    ###################################

    #tf.constant
    # att_high_level  = Reshape(target_shape=(1,4))(att_high_level)
    #########################
    all_lstm = Concatenate(axis=1)([att_high_level, lstm3])
    # knn_out = Concatenate(axis=2)([knn_tensor, all_lstm])
    #########################
    Dense1_output = Dense(units=5, name='Dense_1')(all_lstm)
    Dense2_output = Dense(units=5, name='Dense_2')(Dense1_output)
    # Dense1_output = Reshape(target_shape=(1,10))(Dense1_output)
    # Dense2 = Concatenate(axis=-1)([Dense1_output, knn_tensor_input])
    # print(Dense2)
    pred = Dense(units=1, name='Dense_3')(Dense2_output)

    # input = Concatenate(axis=1)(input_1+input_2+input_3)

    # input = input_1 + input_2 + input_3
    model = Model(inputs=[input_1[0],input_1[1],input_1[2],input_1[3],input_1[4],input_2[0],input_2[1],input_2[2],input_2[3],input_2[4],input_3[0],input_3[1],input_3[2],input_3[3],input_3[4]], outputs=pred)
    model.compile(optimizer='adagrad', loss='mae')
    return model
#
# out = model(final_data)
# print(out)

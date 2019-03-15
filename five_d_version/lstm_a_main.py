from keras.callbacks import ReduceLROnPlateau
import numpy as np
from five_d_version import data_loader
from five_d_version import knn_new
from five_d_version import models


# def data_reforme(train_x):
#     train_x = train_x.reshape(3, 15, 1, 9, 5)
#     train_x_re = []
#     new_train_x = []
#     for i in range(len(train_x)):
#         new_train_x = []
#         for j in range(len(train_x[i])):
#             new_train_x.append(train_x[i][j])
#         train_x_re.append(new_train_x)
#     return train_x_re
model_path = '/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/model/'

def normal_run():
    day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0, day_1_c1, day_1_c2, \
    day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4, y = data_loader.train_data()

    Knn = np.load('/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/knn.npy')
    Knn_train = data_loader.knn_train_data(Knn)
    print(Knn_train)
    # train_x = data_reforme(train_x)
    model = models.model()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    # print(model)

    for i in range(0, 10):
        print(i, '######################################################################################')
        model.fit(x=[day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0,
                     day_1_c1, day_1_c2, day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4], y=[y],
                  batch_size=3, epochs=100, verbose=2)  # ,callbacks=[reduce_lr]
    # save_model(model, 'model.h5')
    model.save_weights('model.h5')
    # print(model.summary())
    Knn_test = data_loader.knn_test_data(Knn)
    print('########################################')
    day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0, day_1_c1, day_1_c2, \
    day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4, y = data_loader.test_data()

    pred = model.predict(x=[day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0,
                            day_1_c1, day_1_c2, day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4],
                         batch_size=3, verbose=2)
    for i in range(0, 3):
        print(pred[i], y[i])

def eval_model():
    eval_list = []
    for i in range(0,10):
        day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0, day_1_c1, day_1_c2, \
        day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4, y = data_loader.train_data()

        Knn = np.load('/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/knn.npy')
        Knn_train = data_loader.knn_train_data(Knn)
        # print(Knn_train)
        # train_x = data_reforme(train_x)
        model = models.model()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # print(model)

        for i in range(0, 10):
            # print(i, '######################################################################################')
            model.fit(x=[day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0,
                         day_1_c1, day_1_c2, day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4],
                      y=[y], batch_size=3, epochs=100, verbose=0)  # ,callbacks=[reduce_lr]
        # save_model(model, 'model.h5')
        model.save_weights('model.h5')
        # print(model.summary())
        Knn_test = data_loader.knn_test_data(Knn)
        # print('########################################')
        day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0, day_1_c1, day_1_c2, \
        day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3, day_2_c4, y = data_loader.test_data()

        pred = model.predict(x=[day_0_c0, day_0_c1, day_0_c2, day_0_c3, day_0_c4, day_1_c0,
                                day_1_c1, day_1_c2, day_1_c3, day_1_c4, day_2_c0, day_2_c1, day_2_c2, day_2_c3,
                                day_2_c4],
                             batch_size=3, verbose=1)
        for i in range(0, 3):
            print(pred[i], y[i])
        temp_result_list = []
        for l in range(0, 3):
            temp_result_list.append(np.abs(pred[l] - y[l])/np.abs(y[l]))
        temp_result = knn_new.list_sum(temp_result_list) / len(temp_result_list)
        eval_list.append(temp_result)
        print(temp_result)
    eval_result = knn_new.list_sum(eval_list) / len(eval_list)
    print(eval_result)



if __name__ == '__main__':
    eval_model()
    #normal_run()

from keras.callbacks import ReduceLROnPlateau
import numpy as np
from two_d_version import data_loader
from two_d_version import models_addconstant_2d
from two_d_version import knn_new



model_path = '/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/model/'


def test_weight():
    all_result = []
    for i in range(1, 100, 5):
        temp_weight_list = []
        for j in range(0, 3):
            day_0_c0, day_0_c1, day_1_c0, day_1_c1, day_2_c0, day_2_c1, y = data_loader.train_data()
            weight = i / 100
            Knn = knn_new.loop_weight(0.5)
            Knn_train = data_loader.knn_train_data(Knn)
            # train_x = data_reforme(train_x)
            model = models_addconstant_2d.model()
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
            # print(model)

            for k in range(0, 100):
                # print(k,'######################################################################################')
                model.fit(x=[day_0_c0, day_0_c1, day_1_c0,
                             day_1_c1, day_2_c0, day_2_c1,
                             Knn_train], y=[y], batch_size=3, epochs=100, verbose=0)  # ,callbacks=[reduce_lr]

            Knn_test = data_loader.knn_test_data(Knn)
            # print('########################################')
            day_0_c0, day_0_c1, day_1_c0, day_1_c1, day_2_c0, day_2_c1, y = data_loader.test_data()

            pred = model.predict(x=[day_0_c0, day_0_c1, day_1_c0,
                                    day_1_c1, day_2_c0, day_2_c1, Knn_test],
                                 batch_size=3, verbose=2)
            print(pred)
            print(y)
            temp_result_list = []
            for l in range(0, 3):
                temp_result_list.append(pred[l] - y[l])
            temp_result = knn_new.list_sum(temp_result_list) / len(temp_result_list)
            temp_weight_list.append(temp_result)
        temp_weigh_result = knn_new.list_sum(temp_weight_list) / len(temp_weight_list)
        print('weight:', i, 'result_mean', temp_weigh_result)


def one_test():
    temp_weight_list = []
    for j in range(0, 10):
        day_0_c0, day_0_c1, day_1_c0, day_1_c1, day_2_c0, day_2_c1, y = data_loader.train_data()
        Knn = knn_new.loop_weight(0.5)
        Knn_train = data_loader.knn_train_data(Knn)
        # train_x = data_reforme(train_x)
        model = models_addconstant_2d.model()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # print(model)

        for k in range(0, 10):
            # print(k,'######################################################################################')
            model.fit(x=[day_0_c0, day_0_c1, day_1_c0,day_1_c1, day_2_c0, day_2_c1,Knn_train], y=[y], batch_size=3, epochs=100, verbose=0)  # ,callbacks=[reduce_lr]

        Knn_test = data_loader.knn_test_data(Knn)
        # print('########################################')
        day_0_c0, day_0_c1, dday_1_c0, day_1_c1, day_2_c0, day_2_c1, y = data_loader.test_data()

        pred = model.predict(x=[day_0_c0, day_0_c1, day_1_c0,
                                day_1_c1, day_2_c0, day_2_c1, Knn_test],
                             batch_size=3, verbose=2)
        print(pred)
        print(y)
        temp_result_list = []
        for l in range(0, 3):
            temp_result_list.append(np.abs(pred[l] - y[l])/np.abs(y[l]))
        temp_result = knn_new.list_sum(temp_result_list) / len(temp_result_list)
        temp_weight_list.append(temp_result)
        print(temp_weight_list)
    temp_weigh_result = knn_new.list_sum(temp_weight_list) / len(temp_weight_list)
    print(temp_weigh_result)
if __name__ == '__main__':
    one_test()
    # test_weight()

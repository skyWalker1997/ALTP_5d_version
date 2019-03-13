import numpy as np
from sklearn import preprocessing

np_path = '/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/'


def get_np():
    # down_len = np.load(np_path+'down_len.npy').reshape(10,9,6)
    # up_len = np.load(np_path+'up_len.npy').reshape(10,9,6)
    down_up_count = np.load(np_path+'down_up_count.npy').reshape(10,9,6)
    # speed = np.load(np_path+'speed.npy').reshape(10,9,6)
    driver = np.load(np_path+'driver.npy').reshape(10,9,6)
    # all = np.load(np_path+'all.npy')
    # down_len = down_len.reshape(10, 3, 3, 6)
    # up_len = up_len.reshape(10, 3, 3, 6)
    down_up_count =  down_up_count.reshape(10, 3, 3, 6)
    # speed = speed.reshape(10, 3, 3, 6)
    driver = driver.reshape(10, 3, 3, 6)
    return down_up_count,driver

def get_stand(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = data.reshape(9,5)
    data = min_max_scaler.fit_transform(data)
    data = np.array((data)).reshape(3,3,5)
    return data

def y_get_stand(data):
    min_max_scaler = preprocessing.MinMaxScaler()


def get_train_data(down_up_count,driver):
    # day_0_c0 = []
    # day_0_c1 = []
    day_0_c2 = []
    day_0_c3 = []
    # day_0_c4 = []
    # day_1_c0 = []
    # day_1_c1 = []
    day_1_c2 = []
    day_1_c3 = []
    # day_1_c4 = []
    # day_2_c0 = []
    # day_2_c1 = []
    day_2_c2 = []
    day_2_c3 = []
    # day_2_c4 = []
    y = []
    for i in range(0,5):
        # day_0_c0.append(np.array((down_len[i,:,:,:5])))
        # day_0_c1.append(np.array((up_len[i,:,:,:5])))
        day_0_c2.append(np.array((down_up_count[i,:,:,:5])))
        day_0_c3.append(np.array((driver[i,:,:,:5])))
        # day_0_c4.append(np.array((speed[i,:,:,:5])))
        # day_1_c0.append(np.array((down_len[i+1,:,:,:5])))
        # day_1_c1.append(np.array((up_len[i+1,:,:,:5])))
        day_1_c2.append(np.array((down_up_count[i+1,:,:,:5])))
        day_1_c3.append(np.array((driver[i+1,:,:,:5])))
        # day_1_c4.append(np.array((speed[i+1,:,:,:5])))
        # day_2_c0.append(np.array((down_len[i+2,:,:,:5])))
        # day_2_c1.append(np.array((up_len[i+2,:,:,:5])))
        day_2_c2.append(np.array((down_up_count[i+2,:,:,:5])))
        day_2_c3.append(np.array((driver[i+2,:,:,:5])))
        # day_2_c4.append(np.array((speed[i+2,:,:,:5])))
        y.append([down_up_count[i+2,1,1,5]])
    return day_0_c2, day_0_c3, day_1_c2, day_1_c3, day_2_c2, day_2_c3,y


def get_test_data(down_up_count,driver):
    # day_0_c0 = []
    # day_0_c1 = []
    day_0_c2 = []
    day_0_c3 = []
    # day_0_c4 = []
    # day_1_c0 = []
    # day_1_c1 = []
    day_1_c2 = []
    day_1_c3 = []
    # day_1_c4 = []
    # day_2_c0 = []
    # day_2_c1 = []
    day_2_c2 = []
    day_2_c3 = []
    # day_2_c4 = []
    y = []
    for i in range(5, 8):
        # day_0_c0.append(np.array((down_len[i, :, :, :5])))
        # day_0_c1.append(np.array((up_len[i, :, :, :5])))
        day_0_c2.append(np.array((down_up_count[i, :, :, :5])))
        day_0_c3.append(np.array((driver[i, :, :, :5])))
        # day_0_c4.append(np.array((speed[i, :, :, :5])))
        # day_1_c0.append(np.array((down_len[i + 1, :, :, :5])))
        # day_1_c1.append(np.array((up_len[i + 1, :, :, :5])))
        day_1_c2.append(np.array((down_up_count[i + 1, :, :, :5])))
        day_1_c3.append(np.array((driver[i + 1, :, :, :5])))
        # day_1_c4.append(np.array((speed[i + 1, :, :, :5])))
        # day_2_c0.append(np.array((down_len[i + 2, :, :, :5])))
        # day_2_c1.append(np.array((up_len[i + 2, :, :, :5])))
        day_2_c2.append(np.array((down_up_count[i + 2, :, :, :5])))
        day_2_c3.append(np.array((driver[i + 2, :, :, :5])))
        # day_2_c4.append(np.array((speed[i + 2, :, :, :5])))
        y.append([down_up_count[i + 2, 1, 1, 5]])
    return day_0_c2, day_0_c3, day_1_c2, day_1_c3, day_2_c2, day_2_c3, y


def test_data():
    down_up_count,driver= get_np()
    day_0_c0, day_0_c1,day_1_c0, day_1_c1,day_2_c0, day_2_c1, y = get_test_data(down_up_count, driver)
    return day_0_c0, day_0_c1, day_1_c0, day_1_c1, day_2_c0, day_2_c1, y


def train_data():
    down_up_count,driver = get_np()
    day_0_c0, day_0_c1, day_1_c0, day_1_c1, day_2_c0, day_2_c1,y= get_train_data(down_up_count, driver)
    return day_0_c0, day_0_c1, day_1_c0, day_1_c1, day_2_c0, day_2_c1,y

def knn_train_data(knn_data):
    train_data = []
    for i in range(0,5):
        temp = np.array(knn_data[i]).reshape(1,5)
        train_data.append(temp)
    return train_data

def knn_test_data(knn_data):
    test_data = []
    for i in range(5, 8):
        temp = np.array(knn_data[i]).reshape(1, 5)
        test_data.append(temp)
    return test_data

import re
from os import listdir
import json as json
import numpy as np
import operator

from sklearn import preprocessing

P_file = '/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/origin_data/P_volume_data/'
Out_file = '/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/'


date_list = ['0416','0417','0418','0419','0420','0421','0422','0423','0424','0425']
time_slot_list = ['0901','0902','1001','1002','1101']


def get_geo_list():
    grid_list = []
    j = 100
    while (j < 2100):
        for i in range(1, 11):
            if j < 1000:
                grid_list.append('0' + str(j + i))
            else:
                grid_list.append(str(j + i))
        j += 100
    return grid_list



def read_data_in_line(DATA_PATH):
    data = []
    for line in open(DATA_PATH,'r',encoding='utf-8'): #设置文件对象并读取每一行文件
        line = line[:-1]
        data.append(line)               #将每一行文件加入到list中
    return data

def read_P_file(geo_list):
    date_up_len_list = []
    date_down_len_lsit = []
    date_up_count_list = []
    date_down_count_list = []
    for date in date_list:
        geo_up_len_lsit = []
        geo_down_len_lsit = []
        geo_up_count_list = []
        geo_down_count_list = []
        for geo_slot in geo_list:
            down_data = read_data_in_line(P_file + 'P_' + date + '_' + geo_slot + '_down.txt')
            up_data = read_data_in_line(P_file + 'P_' + date + '_' + geo_slot + '_up.txt')
            up_count, down_count, down_len, up_len = P_file_processor(down_data, up_data, date, time_slot_list)
            geo_up_len_lsit.append(up_len)
            geo_down_len_lsit.append(down_len)
            geo_up_count_list.append(up_count)
            geo_down_count_list.append(down_count)
        date_up_len_list.append(geo_up_len_lsit)
        date_down_len_lsit.append(geo_down_len_lsit)
        date_up_count_list.append(geo_up_count_list)
        date_down_count_list.append(geo_down_count_list)
    return  date_up_count_list, date_down_count_list, date_up_len_list, date_down_len_lsit


def P_file_processor(down_data, up_data, date, time_slot_list):
    up_count, down_count = get_up_down_count(down_data, up_data, date, time_slot_list,solo_flag=0)
    down_len = get_down_len(down_data, date, time_slot_list)
    up_len = get_up_len(up_data, date, time_slot_list)
    return up_count, down_count, down_len, up_len



def get_up_down_count(down_data,up_data,date,time_slot_list,solo_flag):
    down_conut = []
    up_count = []
    for data_in_line in down_data:
        down = re.split('\t',data_in_line)
        if down[0].replace(date,'') in time_slot_list:
            down_conut.append(float(down[1]))
        else:
            pass
    for data_in_line in up_data:
        up = re.split('\t',data_in_line)
        if up[0].replace(date,'') in time_slot_list:
            up_count.append(float(up[1]))
        else:
            pass

    if solo_flag == 1:
        count = []
        for i in range(len(up_count)):
            count.append(down_conut[i]-up_count[i])
        return count
    else:
        return up_count,down_conut

def get_up_len(up_data,date,time_slot_list):
    up = []
    for data_in_line in up_data:
        up_len = re.split('\t', data_in_line)
        if up_len[0].replace(date, '') in time_slot_list:
            len_str = up_len[2].replace('\'','\"')
            len_js = json.loads(len_str)
            len = len_js['15+'][0]*len_js['15+'][1]+len_js['15'][0]*len_js['15'][1]+len_js['10'][0]*len_js['10'][1]+len_js['5'][0]*len_js['5'][1]
            up.append(float(len))
    return up


def get_down_len(down_data,date,time_slot_list):
    down = []
    for data_in_line in down_data:
        down_len = re.split('\t', data_in_line)
        if down_len[0].replace(date, '') in time_slot_list:
            len_str = down_len[2].replace('\'', '\"')
            len_js = json.loads(len_str)
            len = len_js['15+'][0] * len_js['15+'][1] + len_js['15'][0] * len_js['15'][1] + len_js['10'][0] * \
                  len_js['10'][1] + len_js['5'][0] * len_js['5'][1]
            down.append(float(len))
    return down


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 根据欧式距离计算训练集中每个样本到测试点的距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 计算完所有点的距离后，对数据按照从小到大的次序排序
    sortedDistIndicies = distances.argsort()
    # 确定前k个距离最小的元素所在的主要分类，最后返回发生频率最高的元素类别
    classCount = {}
    distance_slot_index = []
    distance_value = []
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        distance_slot_index.append(sortedDistIndicies[i])
        distance_value.append(distances[sortedDistIndicies[i]])
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    knn_geo = [sortedClassCount[1][0],sortedClassCount[2][0],sortedClassCount[3][0]]
    return knn_geo, distance_slot_index[1:],distance_value[1:]

def get_one_day_data(date,geo_list):
    one_day_data = []
    for geo in geo_list:
        down_data = read_data_in_line(P_file + 'P_' + date + '_' + geo + '_down.txt')
        up_data = read_data_in_line(P_file + 'P_' + date + '_' + geo + '_up.txt')
        for i in range(len(time_slot_list)):
            one_day_data.extend(get_up_down_count(down_data,up_data,date,time_slot_list[i],1))
    return one_day_data

def get_mul_geo_day_data(date,geo_list):
    mul_geo_data = []
    for geo in geo_list:
        down_data = read_data_in_line(P_file + 'P_' + date + '_' + geo + '_down.txt')
        up_data = read_data_in_line(P_file + 'P_' + date + '_' + geo + '_up.txt')
        temp_one_geo = []
        for i in range(len(time_slot_list)):
            temp_one_geo.extend(get_up_down_count(down_data, up_data, date, time_slot_list[i], 1))
        mul_geo_data.append(temp_one_geo)
    return mul_geo_data



def calculate_final_value(distance,down_up_count,target_down_up_count):
    distance_sum = list_sum(distance)
    final_value = 0
    same_self_param = []
    for i in range(len(distance)):
        same_self_param.append(distance[i]/distance_sum)
    for j in range(len(distance)):
        final_value = final_value+down_up_count[j]*same_self_param[j]
    return final_value

def list_sum(list_data):
    sum = 0
    for i in range(len(list_data)):
        sum = sum+list_data[i]
    return sum


def get_auto_corr(timeSeries,i = 4):
    '''
     Descr:输入：时间序列timeSeries，滞后阶数k
               输出：时间序列timeSeries的k阶自相关系数
          l：序列timeSeries的长度
          timeSeries1，timeSeries2:拆分序列1，拆分序列2
          timeSeries_mean:序列timeSeries的均值
          timeSeries_var:序列timeSeries的每一项减去均值的平方的和
    '''
    auto_corr_list = []
    for k in range(1,i+1):
        l = len(timeSeries)
        #取出要计算的两个数组
        timeSeries1 = timeSeries[0:l-k]
        timeSeries2 = timeSeries[k:]
        timeSeries_mean = np.mean(timeSeries)
        timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
        auto_corr = 0
        for i in range(l-k):
            if timeSeries_var == 0:
                temp =0
                auto_corr = auto_corr+temp
            else:
                temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
                auto_corr = auto_corr + temp
        auto_corr_list.append(auto_corr)
    # print(auto_corr_list)
    return auto_corr_list


def get_all_gac(one_day_np):
    '''
    计算一天所有区域的自相关系数值
    '''
    all_geo_gac = []
    for i in range(0,200):
        one_geo_gac = []
        for j in range(0,4):
            temp = get_auto_corr(one_day_np[i,j])
            one_geo_gac.append(temp)
        all_geo_gac.append(one_geo_gac)
    return all_geo_gac

def weight_calculate(data,weight):
    calc_data = []
    demention_len = len(data.shape)
    if demention_len == 1:
        calc_data.extend(data[:8]*weight)
        calc_data.extend(data[8:]*(1-weight))
    else:
        for i in range(len(data)):
            temp_calc_data = []
            temp_calc_data.extend(data[i,:8]*weight)
            temp_calc_data.extend(data[i,8:]*(1-weight))
            calc_data.append(temp_calc_data)
    return calc_data


def loop_weight(weight):
    geo_list = get_geo_list()
    date_up_count_list, date_down_count_list, date_up_len_list, date_down_len_lsit = read_P_file(geo_list)
    data_all_np = np.array((date_up_count_list,date_down_count_list,date_up_len_list,date_down_len_lsit))
    data_all_np = np.swapaxes(data_all_np, 0, 1)
    data_all_np = np.swapaxes(data_all_np, 1, 2)
    data_gac_np = []
    for i in range(2,10):
        data_gac_np.append(get_all_gac(data_all_np[i]))
    data_gac_np = np.array(data_gac_np)
    final_value = []
    for i in range(0, 8):
        one_day = data_gac_np[i, 92, :, :].reshape(16)
        all_data = data_gac_np[i].reshape(200,16)
        one_day = weight_calculate(one_day,weight)
        all_data = weight_calculate(all_data,weight)
        one_day = np.array(one_day)
        all_data = np.array(all_data)
        knn_geo, distance_slot_index, distance_value = classify(one_day, all_data, geo_list, 4)
        one_day_down_up_count = get_one_day_data(date_list[i], ['1003'])
        knn_day_down_up_count = get_mul_geo_day_data(date_list[i],knn_geo)
        knn_day_down_up_count = np.array(knn_day_down_up_count)
        knn_day_down_up_count = np.swapaxes(knn_day_down_up_count,0,1)
        one_day_value = []
        for i in range(0,5):
            one_time_value = calculate_final_value(distance_value,knn_day_down_up_count[i],one_day_down_up_count[i])
            one_day_value.append(one_time_value)
        final_value.append(one_day_value)
    final_value  = get_stand(final_value)
    return final_value



def get_stand(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data

if __name__ == '__main__':

    loop_weight(0.3)
    # geo_list = get_geo_list()
    # date_up_count_list, date_down_count_list, date_up_len_list, date_down_len_lsit = read_P_file(geo_list)
    # data_all_np = np.array((date_up_count_list,date_down_count_list,date_up_len_list,date_down_len_lsit))
    # data_all_np = np.swapaxes(data_all_np, 0, 1)
    # data_all_np = np.swapaxes(data_all_np, 1, 2)
    # data_gac_np = []
    # for i in range(2,10):
    #     data_gac_np.append(get_all_gac(data_all_np[i]))
    #
    # data_gac_np = np.array(data_gac_np)
    # print(data_gac_np.shape)
    #
    # final_value = []
    # for i in range(0,8):
    #     one_day = data_gac_np[i,92,:,:].reshape(1,16)
    #     all_data = data_gac_np[i].reshape(200,16)
    #
    #     knn_geo, distance_slot_index, distance_value = classify(one_day,all_data,geo_list,4)
    #     # print(knn_geo,distance_value)
    #     one_day_down_up_count = get_one_day_data(date_list[i],['1003'])
    #     knn_day_down_up_count = get_mul_geo_day_data(date_list[i],knn_geo)
    #     knn_day_down_up_count = np.array(knn_day_down_up_count)
    #     knn_day_down_up_count = np.swapaxes(knn_day_down_up_count,0,1)
    #     one_day_value = []
    #     for i in range(0,5):
    #         one_time_value = calculate_final_value(distance_value,knn_day_down_up_count[i],one_day_down_up_count[i])
    #         one_day_value.append(one_time_value)
    #     final_value.append(one_day_value)
    #     print('################################')
    # np.save(Out_file+'knn',final_value)
    #
    #

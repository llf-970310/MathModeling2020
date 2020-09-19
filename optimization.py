from pprint import pprint

import joblib
import pandas
import numpy as np

from data_processing import get_variable_range


def greedy(sample_dict: dict, initial_ron: float, row_index: int):
    print(initial_ron)
    range_data = get_variable_range("/Users/faye/Downloads/数模题/附件四：354个操作变量信息.xlsx")
    mapping_data = pandas.read_excel("./data/title.xlsx")
    variable_list = []
    for key, val in sample_dict.items():
        variable_list.append(key)
    variable_len = len(variable_list)

    count = 0
    while True:
        # 随机挑选一个变量进行调整
        random = np.random.randint(low=0, high=variable_len)
        variable_pick_no = variable_list[random]
        # index 1-14 为非操作变量，无法调整
        if 1 <= variable_pick_no <= 14:
            continue
        variable_pick = mapping_data[variable_pick_no][1]
        # print("pick_index: %d, pick item: %s" % (variable_pick_no, variable_pick))

        # 获取该变量的相关信息
        low = range_data[variable_pick]["lower_bound"]
        high = range_data[variable_pick]["upper_bound"]
        delta = range_data[variable_pick]["delta"]
        cur_val = sample_dict[variable_pick_no]

        # 获取变化后的数据
        new_sample = sample_dict
        new_sample[variable_pick_no] = cur_val + delta
        new_ron = get_new_ron(new_sample, row_index)

        # 变更完满足如下条件：
        # 1. 变量仍在范围内
        # 2. 硫含量小于 5
        # 3. ron 损失有降低
        if low <= new_sample[variable_pick_no] <= high and check_sulfur(new_sample) is True and new_ron < initial_ron:
            initial_ron = new_ron
            sample_dict = new_sample
        else:
            count += 1

        if count > 10:
            break

    print(initial_ron)


# 判断硫含量
def check_sulfur(sample_dict: dict):
    model = joblib.load("./classification_model.pkl")
    df_x = pandas.read_excel("./data/x2.xlsx")

    for key, val in sample_dict.items():
        if key in df_x.columns:
            df_x.loc[0, key] = val

    return model.predict(df_x[0:1])[0] == 1.0


# 判断 ron 损失
def get_new_ron(sample_dict: dict, index: int):
    model = joblib.load("./regression_model.pkl")
    df_x = pandas.read_excel("./data/x1.xlsx")

    for key, val in sample_dict.items():
        df_x.loc[index, key] = val

    return model.predict(df_x[index:index + 1])[0]


if __name__ == "__main__":
    df_x = pandas.read_excel("./data/x1.xlsx")
    df_y = pandas.read_excel("./data/y1.xlsx")

    # 将 df 转为字典数组，列为 key
    record_list = df_x.to_dict('records')

    for i in range(325):
        greedy(record_list[i], df_y.loc[i], i)

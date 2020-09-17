from pprint import pprint

import pandas


# 读附件四，获取变量的范围
def get_variable_range(path: str) -> dict:
    data = pandas.read_excel(path)
    range_dict = {}
    for index, row in data.iterrows():
        var_id, var_name, var_range = row[1], row[2], row[3]
        # 范围去括号
        var_range = var_range.replace('(', '')
        var_range = var_range.replace(')', '')
        var_range = var_range.replace('（', '')
        var_range = var_range.replace('）', '')
        # 存在负数情况
        split_list = var_range.split("-")
        range_dict[var_id] = {}
        range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = 0, 0
        # left + right +
        if len(split_list) == 2:
            range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = float(split_list[0]), float(
                split_list[1])
        # left - right +
        elif len(split_list) == 3:
            range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = -float(split_list[1]), float(
                split_list[2])
        # left - right -
        elif len(split_list) == 4:
            range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = -float(split_list[1]), -float(
                split_list[3])
        else:
            print("can't handle, range is %s" % var_range)
    # pprint(range_dict)
    return range_dict


# 读附件三，对 285 和 313 数据进行处理
def process_original_data(path: str, sample_no: str, range_dict: dict):
    excel_data = None
    if sample_no == '285':
        # nrows 不包含表头
        excel_data = pandas.read_excel(path, header=[1, 2], sheet_name=4, nrows=40)
    elif sample_no == '313':
        excel_data = pandas.read_excel(path, header=[1, 2], sheet_name=4, skiprows=range(3, 44), nrows=40)

    # 遍历列，col_name 为列名，col_content 为该列的具体内容
    for col_name, col_content in excel_data.iteritems():
        # print("col_name: " + col_name[0])
        if col_name[0] == '时间':
            continue

        # (1)删掉全为 0 或多数为 0 的列
        count = 0
        # 遍历该列的每一行
        for row_index in range(len(col_content)):
            # print("row_index: " + str(row_index))
            # print("row_content: " + str(col_content[row_index]))
            if float(col_content[row_index]) == 0:
                count += 1
        # 超过一半为 0，即删掉该点位
        if count > 40 * 0.5:
            print("delete column " + col_name[0])
            excel_data = excel_data.drop(col_name[0], axis=1)
            continue

        # (4)数据范围筛选
        range_lower, range_upper = range_dict[col_name[0]]['lower_bound'], range_dict[col_name[0]]['upper_bound']
        for row_index in range(len(col_content)):
            if float(col_content[row_index]) < range_lower or float(col_content[row_index]) > range_upper:
                print("var_id: %s, lower: %f, upper: %f" % (col_name[0], range_lower, range_upper))
                print("delete row " + str(row_index))
                # excel_data = excel_data.drop([row_index], axis=0)

        # (5)拉伊达准则去除异常值

    # print(excel_data)


if __name__ == '__main__':
    range_data = get_variable_range("/Users/faye/Downloads/数模题/附件四：354个操作变量信息.xlsx")
    # process_original_data("D:\\Downloads\\2020年中国研究生数学建模竞赛赛题\\2020年B题\\数模题\\附件三：285号和313号样本原始数据.xlsx")
    process_original_data("/Users/faye/Downloads/数模题/附件三：285号和313号样本原始数据.xlsx", "285", range_data)

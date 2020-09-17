import pandas


# 读附件四，获取变量的范围
def get_variable_range(path: str) -> dict:
    data = pandas.read_excel(path)


# 读附件三，对 285 和 313 数据进行处理
def process_original_data(path: str):
    original_data = pandas.read_excel(path, header=1, sheet_name=4)
    # 遍历列，index为header，col为该列的具体内容
    for col_name, col_content in original_data.head(3).iteritems():
        print("col_name: " + col_name)
        for row_index in range(len(col_content)):
            print("row_index: " + str(row_index))
            print("row_content: " + str(col_content[row_index]))


if __name__ == '__main__':

    process_original_data("D:\\Downloads\\2020年中国研究生数学建模竞赛赛题\\2020年B题\\数模题\\附件三：285号和313号样本原始数据.xlsx")

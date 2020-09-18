import pandas
import matplotlib.pyplot as plt
from sklearn import tree, linear_model, svm, neighbors, ensemble
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import ExtraTreeRegressor


def try_different_method(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


def modeling(path: str):
    excel_data = pandas.read_excel(path, header=0)
    # 将特征划分到 X 中，标签划分到 Y 中
    x = excel_data.iloc[:, 0:21]
    y = excel_data['ret']
    # 使用train_test_split函数划分数据集(训练集占75%，测试集占25%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 决策树
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

    # 线性回归
    model_LinearRegression = linear_model.LinearRegression()

    # SVM
    model_SVR = svm.SVR()

    # KNN
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()

    # 随机森林
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树

    # Adaboost
    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树

    # GBRT
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树

    # Bagging回归
    model_BaggingRegressor = BaggingRegressor()

    # ExtraTree极端随机树回归
    model_ExtraTreeRegressor = ExtraTreeRegressor()

    try_different_method(model_BaggingRegressor, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    modeling("/Users/faye/Downloads/variable.xlsx")

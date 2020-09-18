import joblib
import pandas
import matplotlib.pyplot as plt
from sklearn import tree, linear_model, svm, neighbors, ensemble, metrics
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.tree import ExtraTreeRegressor


def get_model(model_name):
    model_dict = {
        # 回归
        "model_DecisionTreeRegressor": tree.DecisionTreeRegressor(),  # 决策树
        "model_LinearRegression": linear_model.LinearRegression(),  # 线性回归
        "model_SVR": svm.SVR(),  # SVM
        "model_KNeighborsRegressor": neighbors.KNeighborsRegressor(),  # KNN
        "model_RandomForestRegressor": ensemble.RandomForestRegressor(n_estimators=20),  # 随机森林，这里使用20个决策树
        "model_AdaBoostRegressor": ensemble.AdaBoostRegressor(n_estimators=50),  # Adaboost，这里使用50个决策树
        "model_GradientBoostingRegressor": ensemble.GradientBoostingRegressor(n_estimators=100),  # GBRT，这里使用100个决策树
        "model_BaggingRegressor": BaggingRegressor(),  # Bagging回归
        "model_ExtraTreeRegressor": ExtraTreeRegressor(),  # ExtraTree极端随机树回归
        # 分类
        "model_LogisticRegression": LogisticRegression(C=1000, class_weight={0: 0.8, 1: 0.2}),  # 逻辑回归
        "model_SVC": svm.SVC(class_weight="balanced"),  # 向量机
        "model_RandomForestClassifier": RandomForestClassifier(n_estimators=7, class_weight="balanced")  # 随机森林
    }

    return model_dict[model_name]


# 读取数据并划分训练集和测试集
def read_data(path: str, type: str):
    # excel_data = pandas.read_excel(path, header=0)
    # # 将特征划分到 X 中，标签划分到 Y 中
    # x = excel_data.iloc[:, 0:15]
    # y = excel_data['ret']
    # # 使用train_test_split函数划分数据集(训练集占75%，测试集占25%)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 特征
    df_x = pandas.read_excel("./jrb/x.xlsx")
    # 标签
    df_y = pandas.read_excel("./y.xlsx")
    x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y[type].values, test_size=0.3, random_state=0)

    return x_train, x_test, y_train, y_test


def fit_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()

    return model, result


# 二分类（离散数据），针对于硫含量
def classification(model, x_train, x_test, y_train, y_test):
    # 训练模型，并返回训练后的模型和预测结果集
    model, result = fit_model(model, x_train, x_test, y_train, y_test)
    # 模型持久化
    joblib.dump(model, 'classification_model.pkl')

    print("精确率：%f" % metrics.precision_score(y_test, result))
    print("召回率：%f" % metrics.recall_score(y_test, result))
    print(metrics.classification_report(y_test, result))


# 回归（连续数据），针对于 ROH
def regression(model, x_train, x_test, y_train, y_test):
    fit_model(model, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    # 回归预测 roh
    x_train, x_test, y_train, y_test = read_data("", "roh")
    model = get_model("model_LinearRegression")
    regression(model, x_train, x_test, y_train, y_test)
    # 二分类预测硫含量
    x_train, x_test, y_train, y_test = read_data("", "s")
    model = get_model("model_LogisticRegression")
    classification(model, x_train, x_test, y_train, y_test)

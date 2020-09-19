from sklearn import tree, linear_model, svm, neighbors, ensemble, metrics
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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


def get_best_param(x_train, y_train):
    params = {'C': [0.0001, 1, 100, 1000],
              'max_iter': [1, 10, 100],
              'class_weight': ['balanced', None, {0: 0.8, 1: 0.2}],
              }
    lr = LogisticRegression()
    clf = GridSearchCV(lr, param_grid=params, cv=10, scoring="average_precision")
    clf.fit(x_train, y_train)
    print(clf.best_params_)
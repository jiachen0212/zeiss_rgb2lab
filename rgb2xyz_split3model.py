# coding=utf-8
'''
rgb2xyz, 拆分成3个model, 分别完成: rgb2x rgb2y rgb2z
xgboost

'''

import json
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import numpy as np
from scipy.stats import uniform, randint
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS


'''
js1_all = dict()
js2_all = dict()
js1 = json.load(open(r'./all_col3.json', 'r'))
js1_1 = json.load(open(r'./all_col3_0821.json', 'r'))
# for k, v in js1.items():
#     if len(v) == 6:
#         print(k)

js2 = json.load(open(r'./0812lab_value1.json', 'r'))
js2_1 = json.load(open(r'./all_lab_value.json', 'r'))

js1_all.update(js1)
js1_all.update(js1_1)
js2_all.update(js2_1)
js2_all.update(js2)

data = json.dumps(js1_all)
with open(r'./all_data_rgb3.json', 'w') as js_file:
    js_file.write(data)

data = json.dumps(js2_all)
with open(r'./all_data_lab.json', 'w') as js_file:
    js_file.write(data)

'''




# 简单用一个xgboost?
# def cross_val(X, y, index):
#     kfold = KFold(n_splits=2, shuffle=True, random_state=42)
#
#     scores = []
#
#     # hyperparameter_searching beat parameters
#     parameters = json.load(open(r'./parameter_{}.json'.format(index), 'r'))
#
#     for train_index, test_index in kfold.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         # xgb_model = xgb.XGBRegressor(objective="reg:linear")
#         # xgb_model = ensemble.RandomForestRegressor(n_estimators=200)
#         xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
#         xgb_model.fit(X_train, y_train)
#
#         y_pred = xgb_model.predict(X_test)
#
#         c, d = 0, 0
#         for index, item in enumerate(y_pred):
#             # print("y_target: {:.3f}, y_predict: {:.3f}, abs_delta: {:.3f}".format(y_test[index], y_pred[index], abs(y_test[index] - y_pred[index])))
#             c += 1
#             if abs(y_test[index] - y_pred[index]) <= 0.5:
#                 d += 1
#         print("test all size: {}, ok size: {}, percent: {}".format(c, d, d/c))
#         scores.append(mean_squared_error(y_test, y_pred))
def cross_val(X, Y, index, green_blue, rgb_ImgName, ff):
    X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3, random_state=88)

    # 查看train中异常样本数量
    for ii, item in enumerate(X_train):
        value = ''.join(str(int(a)) for a in X_train[ii]) + str(y_train[ii])
        key_ = rgb_ImgName[value]
        if key_ in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
            print(key_, '=====')

    scores = []

    # hyperparameter_searching beat parameters
    parameters = json.load(open(r'./parameter_{}_{}.json'.format(index, green_blue), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    c, d = 0, 0
    for index, item in enumerate(y_pred):
        # print("y_target: {:.3f}, y_predict: {:.3f}, abs_delta: {:.3f}".format(y_test[index], y_pred[index], abs(y_test[index] - y_pred[index])))
        c += 1
        if abs(y_test[index] - y_pred[index]) <= 0.5:
            d += 1
        else:
            value = ''.join(str(int(a)) for a in X_test[index]) + str(y_test[index])
            info = str(rgb_ImgName[value]) + '\tdiff: ' + str(abs(y_test[index] - y_pred[index]))
            print(X_test[index], y_test[index], info)
            ff.write(info + '\n')
    print("test all size: {}, ok size: {}, percent: {}".format(c, d, d/c))
    scores.append(mean_squared_error(y_test, y_pred))


def report_best_scores(results, index, green_blue, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            # print("Model with rank: {0}".format(i))
            # print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            #       results['mean_test_score'][candidate],
            #       results['std_test_score'][candidate]))
            parameter = results['params'][candidate]
            # print("Parameters: {0}".format(parameter))

    # 超参落盘
    data = json.dumps(parameter)
    with open(r'./parameter_{}_{}.json'.format(index, green_blue), 'w') as js_file:
        js_file.write(data)



def hyperparameter_searching(X, y, index, green_blue):
    xgb_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, y)

    report_best_scores(search.cv_results_, index, green_blue, 5)



def lab2xyz(l,a,b):
    fy = (l+16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    if np.power(fy, 3) > 0.008856:
        y = np.power(fy, 3)
    else:
        y = (fy - 16 / 116.0) / 7.787

    if np.power(fx, 3) > 0.008856:
        x = np.power(fx, 3)
    else:
        x = (fx - 16 / 116.0) / 7.787

    if np.power(fz, 3) > 0.008856:
        z = np.power(fz, 3)
    else:
        z = (fz - 16 / 116.0) / 7.787
    x *= 94.81211415
    y *= 100
    z *= 107.3369399

    return [x,y,z]


def load_data(json_x, json_y, index, green_blue):
    rgb_ImgName = dict()
    X , Y = [], []
    for k, v in json_x.items():
        dir_index = int(k.split('_')[0])
        if not green_blue:
            if dir_index <= 21 or k in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                X.append([float(a) for a in json_x[k]])
                v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                Y.append(v_[index])
                rgb_ImgName[''.join(a for a in json_x[k]) + str(v_[index])] = k
        else:
            if dir_index > 21 and k not in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                X.append([float(a) for a in json_x[k]])
                v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                Y.append(v_[index])
                rgb_ImgName[''.join(a for a in json_x[k]) + str(v_[index])] = k

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(len(rgb_ImgName))

    return X, Y, rgb_ImgName



if __name__ == "__main__":
    # all data
    js_x = json.load(open(r'./all_data_rgb3.json', 'r'))
    js_y = json.load(open(r'./all_data_lab.json', 'r'))

    # green: 0, blue: 1
    green_blue = 1

    flags = ['x', 'y', 'z']
    ff = open(r'./bad_blue.txt', 'w')
    for i in range(3):
        ff.write("for {}: \n".format(flags[i]))
        print("for {} value".format(flags[i]))
        X, Y, rgb_ImgName = load_data(js_x, js_y, i, green_blue)
        assert X.shape[0] == Y.shape[0]

        # use xgboost
        # hyperparameter_searching(X, Y, i, green_blue)
        cross_val(X, Y, i, green_blue, rgb_ImgName, ff)
        ff.write("\n")

        print('\n')


        # 蓝绿样本划分
        # for k, v in js_x.items():
        #     v = [int(a) for a in v]
        #     dir_index = int(k.split('_')[0])
        #     # rgb 判断g值是否大于b值即可.
        #     if v[1] > v[2]:
        #         if dir_index > 21:
        #             print(k, '==')



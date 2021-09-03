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

def prepare_data():
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



def cross_val(X, Y, index, green_blue, rgb_ImgName):
    single_lab_res = dict()

    X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3, random_state=88)

    # 查看train中异常样本数量
    # for ii, item in enumerate(X_train):
    #     value = ''.join(str(a) for a in X_train[ii])
    #     key_ = rgb_ImgName[value]
    #     if key_ in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
    #         print(key_, '=====')


    # hyperparameter_searching beat parameters
    parameters = json.load(open(r'./rgb2lab_parameter_{}_{}.json'.format(index, green_blue), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X)
    for ii, item in enumerate(y_pred):
        value = ''.join(str(a) for a in X[ii])
        info = rgb_ImgName[value]
        single_lab_res[info] = str(item)

    data = json.dumps(single_lab_res)
    with open(r"./single_lab_{}.json".format(index), 'w') as js_file:
        js_file.write(data)


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
    with open(r'./rgb2lab_parameter_{}_{}.json'.format(index, green_blue), 'w') as js_file:
        js_file.write(data)



def hyperparameter_searching(X, y, index, green_blue):

    if not green_blue:
        xgb_model = xgb.XGBRegressor()
        params = {
            "colsample_bytree": uniform(0.9, 0.1),
            "gamma": uniform(0, 0.5),   # gamma越小, 模型越复杂..
            "learning_rate": uniform(0.01, 0.5),  # default 0.1
            "max_depth": randint(2, 6),  # default 3
            "n_estimators": randint(80, 120),  # default 100
            "subsample": uniform(0.6, 0.4)
        }

    else:
        xgb_model = xgb.XGBRegressor()
        params = {
            "colsample_bytree": uniform(0.9, 0.1),
            "gamma": uniform(0, 0.5),   # gamma越小, 模型越复杂..
            "learning_rate": uniform(0.01, 0.3),  # default 0.1
            "max_depth": randint(2, 6),  # default 3
            "n_estimators": randint(80, 120),  # default 100
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
                r_, g_, b_ = [float(a)/255 for a in json_x[k]]
                X.append([r_, g_, b_])
                Y.append(json_y[k][index])
                rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k
        else:
            if dir_index > 21 and k not in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                r_, g_, b_ = [float(a) / 255 for a in json_x[k]]
                X.append([r_, g_, b_])
                Y.append(json_y[k][index])
                rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(len(rgb_ImgName))

    return X, Y, rgb_ImgName


def xyz2lab(x, y, z):
    x /= 94.81211415
    y /= 100
    z /= 107.3369399
    if y > 0.008856:
        fy = np.power(y, 1/3)
    else:
        fy = 7.787 * y + 16 / 116.0
    if x > 0.008856:
        fx = np.power(x, 1/3)
    else:
        fx = 7.787 * x + 16 / 116.0
    if z > 0.008856:
        fz = np.power(z, 1/3)
    else:
        fz = 7.787 * z + 16 / 116.0
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return [l, a, b]



def split_blueand_green():
    # 蓝绿样本划分
    for k, v in js_x.items():
        v = [int(a) for a in v]
        dir_index = int(k.split('_')[0])
        # rgb 判断g值是否大于b值即可.
        if v[1] > v[2]:
            if dir_index > 21:
                print(k, '==')




def check_lab_res(js_x, js_y, ff):
    x_pred = json.load(open(r'./xyz_0.json', 'r'))
    y_pred = json.load(open(r'./xyz_1.json', 'r'))
    z_pred = json.load(open(r'./xyz_2.json', 'r'))
    ks = list(x_pred.keys())

    c = 0
    bad_l, bad_a, bad_b = 0, 0, 0
    for k in ks:
        real_l, real_a, real_b = js_y[k]
        pre_x, pre_y, pre_z = float(x_pred[k]), float(y_pred[k]), float(z_pred[k])
        pre_l, pre_a, pre_b = xyz2lab(pre_x, pre_y, pre_z)
        real_x, real_y, real_z = lab2xyz(real_l, real_a, real_b)

        if abs(pre_l-real_l) <= 0.5 and abs(pre_a-real_a) <= 0.5 and abs(pre_b-real_b) <= 0.5:
            c += 1

        else:
            print("dir_name: {}".format(k))
            print("rgb: {}".format(js_x[k]))
            print("real lab: {}, pred lab: {}".format(js_y[k], [pre_l, pre_a, pre_b]))
            print("diff_l: {}, diff_a: {}, diff_b:{}".format(abs(pre_l - real_l), abs(pre_a - real_a), abs(pre_b - real_b)))
            print("diff_x: {}, diff_y: {}, diff_z:{}".format(abs(pre_x - real_x), abs(pre_y - real_y), abs(pre_z - real_z)))
            print('\n')

            ff.write("dir_name: {}".format(k) + '\n')
            ff.write("rgb: {}".format(js_x[k]) + '\n')
            ff.write("real lab: {}, pred lab: {}".format(js_y[k], [pre_l, pre_a, pre_b]) + '\n')
            ff.write("diff_l: {}, diff_a: {}, diff_b:{}".format(abs(pre_l - real_l), abs(pre_a - real_a),
                                                             abs(pre_b - real_b)) + '\n')
            ff.write("diff_x: {}, diff_y: {}, diff_z:{}".format(abs(pre_x - real_x), abs(pre_y - real_y),
                                                             abs(pre_z - real_z)) + '\n')
            ff.write('\n')

        if abs(pre_l-real_l) > 0.5:
            bad_l += 1
        if abs(pre_a-real_a) > 0.5:
            bad_a += 1
        if abs(pre_b-real_b) > 0.5:
            bad_b += 1


    print("L A B all diff in  0.5: {}, all data size: {}".format(c, len(ks)))
    ff.write('\n')
    ff.write("bad_L: {}, bad_A:{}, bad_B:{}".format(bad_l, bad_a, bad_b) + '\n')
    ff.write("L A B all diff in  0.5: {}, all data size: {}".format(c, len(ks)) + '\n')



def overfiting(X, Y, index, green_blue):
    dfull = xgb.DMatrix(X, Y)

    param1 = json.load(open(r'./parameter_{}_{}.json'.format(index, green_blue), 'r'))
    num_round = 200

    cvresult1 = xgb.cv(param1, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()


def check_lab_res(js_x, js_y):
    l_pred = json.load(open(r"./single_lab_0.json", 'r'))
    a_pred = json.load(open(r"./single_lab_1.json", 'r'))
    b_pred = json.load(open(r"./single_lab_2.json", 'r'))
    ks = list(l_pred.keys())

    c = 0
    bad_l, bad_a, bad_b = 0, 0, 0
    for k in ks:
        real_l, real_a, real_b = js_y[k]
        pre_l, pre_a, pre_b = float(l_pred[k]), float(a_pred[k]), float(b_pred[k])
        if abs(pre_l-real_l) <= 0.5 and abs(pre_a-real_a) <= 0.5 and abs(pre_b-real_b) <= 0.5:
            c += 1
        else:
            print("img_name: {}, diff_l: {}, diff_a: {}, diff_b: {}".format(k, abs(pre_l-real_l), abs(pre_a-real_a), abs(pre_b-real_b)))

        if abs(pre_l-real_l) > 0.5:
            bad_l += 1
        if abs(pre_a-real_a) > 0.5:
            bad_a += 1
        if abs(pre_b-real_b) > 0.5:
            bad_b += 1

    print("ok size: {}".format(c))
    print("bad l:{}, bad a: {}, bas b: {}".format(bad_l, bad_a, bad_b))


if __name__ == "__main__":

    # merge data1 and data2
    prepare_data()

    js_x = json.load(open(r'./all_data_rgb3.json', 'r'))
    js_y = json.load(open(r'./all_data_lab.json', 'r'))

    # green: 0, blue: 1
    green_blue = 1

    flags = ['x', 'y', 'z']
    txts = ["green", "blue"]
    ff = open(r'./bad_{}.txt'.format(txts[green_blue]), 'w')
    for i in range(3):
        print("for {} value".format(flags[i]))
        X, Y, rgb_ImgName = load_data(js_x, js_y, i, green_blue)
        assert X.shape[0] == Y.shape[0]

        # use xgboost
        hyperparameter_searching(X, Y, i, green_blue)
        overfiting(X, Y, i, green_blue)
        cross_val(X, Y, i, green_blue, rgb_ImgName)

    # compare result
    check_lab_res(js_x, js_y)



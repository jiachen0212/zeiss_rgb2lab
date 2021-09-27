# coding=utf-8
'''
rgb2xyz, 拆分成3个model, 分别完成: rgb2x rgb2y rgb2z
xgboost
'''
from util import cnames

colors = list(cnames.keys())
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


def cross_val(X_train, y_train, X, X_test, index, green_blue, rgb_ImgName):
    xyz_res = dict()
    single_xyz_res = r'./xyz_{}.json'.format(index)

    # 查看train中异常样本数量
    for ii, item in enumerate(X_train):
        value = ''.join(str(a) for a in X_train[ii])
        key_ = rgb_ImgName[value]
        if key_ in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
            print(key_, '=====')

    # hyperparameter_searching beat parameters
    parameters = json.load(open(r'./parameter1_{}_{}.json'.format(index, green_blue), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, y_train)

    # y_pred = xgb_model.predict(X_test)
    # for index, item in enumerate(y_pred):
    #     value = ''.join(str(int(a)) for a in X_test[index]) + str(y_test[index])
    #     info = rgb_ImgName[value]
    #     xyz_res[info] = str(item)

    # test all data
    y_pred = xgb_model.predict(X)
    for ii, item in enumerate(y_pred):
        value = ''.join(str(a) for a in X[ii])
        info = rgb_ImgName[value]
        xyz_res[info] = str(item)

    data = json.dumps(xyz_res)
    with open(single_xyz_res, 'w') as js_file:
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
    with open(r'./parameter1_{}_{}.json'.format(index, green_blue), 'w') as js_file:
        js_file.write(data)


def hyperparameter_searching(X, y, index, green_blue):
    # xgb_model = xgb.XGBRegressor()
    # params = {
    #     "colsample_bytree": uniform(0.9, 0.1),
    #     "gamma": uniform(0, 0.),
    #     "learning_rate": uniform(0.03, 0.3),  # default 0.1
    #     "max_depth": randint(2, 6),  # default 3
    #     "n_estimators": randint(100, 150),  # default 100
    #     "subsample": uniform(0.6, 0.4)
    # }

    # xgb_model = xgb.XGBRegressor()
    # params = {
    #     "colsample_bytree": uniform(0.9, 0.1),
    #     "gamma": uniform(0, 0.),   # gamma越小, 模型越复杂..
    #     "learning_rate": uniform(0.01, 0.5),  # default 0.1
    #     "max_depth": randint(2, 8),  # default 3
    #     "n_estimators": randint(100, 150),  # default 100
    #     "subsample": uniform(0.6, 0.4)
    # }

    xgb_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.9, 0.1),
        "gamma": uniform(0, 0.),  # gamma越小, 模型越复杂..
        "learning_rate": uniform(0.01, 0.5),  # default 0.1
        "max_depth": randint(2, 10),  # default 3
        "n_estimators": randint(80, 150),  # default 100
        "subsample": uniform(0.6, 0.4)

    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, y)

    report_best_scores(search.cv_results_, index, green_blue, 5)


def lab2xyz(l, a, b):
    fy = (l + 16.0) / 116.0
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

    return [x, y, z]


def gamma(a):
    if a > 0.04045:
        a = np.power((a + 0.055) / 1.055, 2.4)
    else:
        a /= 12.92

    return a


def show_rgb_gamma(org_rgb, gammed_rgb, green_blue):
    aa = [0, 1, 2]
    greenblue = ["green", "blue"]
    plt.title(r"{} data: org and  gamma_ed".format(greenblue[green_blue]))
    for ii, rgb in enumerate(org_rgb):
        if ii == 0:
            plt.plot(aa, rgb, color='pink', label='org')
            plt.plot(aa, gammed_rgb[ii], color='cornflowerblue', label='gammaed')
        else:
            plt.plot(aa, rgb, color='pink')
            plt.plot(aa, gammed_rgb[ii], color='cornflowerblue')
    plt.legend()
    plt.show()


def show_b_gamma(org_b_gammaes_b):
    aa = [0, 1]
    plt.title(r"gamma b")
    for ii, b1b2 in enumerate(org_b_gammaes_b):
        plt.scatter(aa, b1b2, color=colors[ii], s=2)
    plt.show()


def load_data(json_x, json_y, index, green_blue, gammaed=False):
    X_dict = dict()
    org_b_gammaes_b = []
    org_rgb = []
    gammed_rgb = []
    rgb_ImgName = dict()
    X, Y = [], []
    for k, v in json_x.items():
        dir_index = int(k.split('_')[0])

        if not green_blue:
            # 绿膜数据处理
            if dir_index <= 21 or k in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                r_, g_, b_ = [float(a) / 255 for a in json_x[k]]
                # 是否 gamma 矫正
                if not gammaed:
                    X.append([r_, g_, b_])
                    v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                    Y.append(v_[index])
                    rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k
                else:
                    org_rgb.append([r_, g_, b_])
                    gamma_r_ = gamma(r_)
                    gamma_g_ = gamma(g_)
                    gamma_b_ = gamma(b_)
                    gammed_rgb.append([gamma_r_, gamma_g_, gamma_b_])
                    X.append([gamma_r_, gamma_g_, gamma_b_])
                    X_dict[k] = [gamma_r_, gamma_g_, gamma_b_]
                    v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                    Y.append(v_[index])
                    rgb_ImgName[''.join(str(a) for a in [gamma_r_, gamma_g_, gamma_b_])] = k

        else:
            # 蓝膜数据处理
            if dir_index > 21 and k not in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                r_, g_, b_ = [float(a) / 255 for a in json_x[k]]
                # 是否 gamma 矫正
                if not gammaed:
                    X.append([r_, g_, b_])
                    X_dict[k] = [r_, g_, b_]
                    v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                    Y.append(v_[index])
                    rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k
                else:
                    # if index == 2:
                    #     # 只对b值进行gamma矫正
                    #     gamma_b_ = gamma(b_)
                    # else:
                    #     gamma_b_ = b_
                    # org_b_gammaes_b.append([b_, gamma_b_])
                    # X.append([r_, g_, gamma_b_])
                    org_rgb.append([r_, g_, b_])
                    # rgb均进行gamma矫正
                    gamma_r_ = gamma(r_)
                    gamma_g_ = gamma(g_)
                    gamma_b_ = gamma(b_)
                    gammed_rgb.append([gamma_r_, gamma_g_, gamma_b_])
                    X.append([gamma_r_, gamma_g_, gamma_b_])
                    X_dict[k] = [gamma_r_, gamma_g_, gamma_b_]
                    v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                    Y.append(v_[index])
                    rgb_ImgName[''.join(str(a) for a in [gamma_r_, gamma_g_, gamma_b_])] = k

    X = np.array(X)
    Y = np.array(Y)
    # print(X.shape)
    # print(len(rgb_ImgName))
    # print(len(X_dict))

    # show_b_gamma(org_b_gammaes_b)
    # show_rgb_gamma(org_rgb, gammed_rgb, green_blue)

    return X, Y, rgb_ImgName, X_dict


def xyz2lab(x, y, z):
    x /= 94.81211415
    y /= 100
    z /= 107.3369399
    if y > 0.008856:
        fy = np.power(y, 1 / 3)
    else:
        fy = 7.787 * y + 16 / 116.0
    if x > 0.008856:
        fx = np.power(x, 1 / 3)
    else:
        fx = 7.787 * x + 16 / 116.0
    if z > 0.008856:
        fz = np.power(z, 1 / 3)
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


def check_lab_res(green_blue, js_x, js_y, ff, X_dict):
    aa = [i for i in range(3)]

    f_bad_l = open(r'./bad_l.txt', 'w')
    f_bad_a = open(r'./bad_a.txt', 'w')
    f_bad_b = open(r'./bad_b.txt', 'w')
    green_all_dict = dict()
    blue_bad_a_dict = dict()
    bad_a_rgb = []
    bad_a_dict = dict()
    bad_b_dict = dict()
    ok_a_rgb = []
    bad_b_rgb = []

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

        if abs(pre_l - real_l) <= 0.5 and abs(pre_a - real_a) <= 0.5 and abs(pre_b - real_b) <= 0.5:
            c += 1
    print("L A B all diff in  0.5: {}, all data size: {}".format(c, len(ks)))
        # else:
        #     print("data: {}, diff l: {}, diff a: {}, diff b: {}".format(k, abs(pre_l - real_l), abs(pre_a - real_a),
        #                                                                 abs(pre_b - real_b)))
        # # green_all_dict[''.join(a+',' for a in js_x[k])] = 1
        # if not green_blue:
        #     blue_bad_a_dict[''.join(str(a) + ',' for a in X_dict[k])] = [abs(pre_a - real_a), k]
        #
        #
        # if abs(pre_l - real_l) > 0.5:
        #     bad_l += 1
        #     f_bad_l.write(k + '\n')
        #     f_bad_l.write("rgb: {}".format(js_x[k]) + '\n')
        #     f_bad_l.write("real lab: {}, pred lab: {}".format(js_y[k], [pre_l, pre_a, pre_b]) + '\n')
        #     f_bad_l.write("diff_l: {}, diff_a: {}, diff_b:{}".format(abs(pre_l - real_l), abs(pre_a - real_a),
        #                                                              abs(pre_b - real_b)) + '\n')
        #
        # if abs(pre_a - real_a) > 0.5:
        #     bad_a += 1
        #     f_bad_a.write(k + '\n')
        #     f_bad_a.write("rgb: {}".format(js_x[k]) + '\n')
        #     f_bad_a.write("real lab: {}, pred lab: {}".format(js_y[k], [pre_l, pre_a, pre_b]) + '\n')
        #     f_bad_a.write("diff_l: {}, diff_a: {}, diff_b:{}".format(abs(pre_l - real_l), abs(pre_a - real_a),
        #                                                              abs(pre_b - real_b)) + '\n')
        #
        #     bad_a_rgb.append([float(a) for a in js_x[k]])
        #
        # else:
        #     ok_a_rgb.append([float(a) for a in js_x[k]])
        #
        # bad_a_dict[''.join(a + ',' for a in js_x[k])] = abs(pre_a - real_a)
        #
        # if abs(pre_b - real_b) > 0.5:
        #     bad_b += 1
        #     f_bad_b.write(k + '\n')
        #     f_bad_b.write("rgb: {}".format(js_x[k]) + '\n')
        #
        #     f_bad_b.write("real lab: {}, pred lab: {}".format(js_y[k], [pre_l, pre_a, pre_b]) + '\n')
        #     f_bad_b.write("diff_l: {}, diff_a: {}, diff_b:{}".format(abs(pre_l - real_l), abs(pre_a - real_a),
        #                                                              abs(pre_b - real_b)) + '\n')
        #
        #     bad_b_rgb.append([float(a) for a in js_x[k]])
        #
        # bad_b_dict[''.join(a + ',' for a in js_x[k])] = abs(pre_b - real_b)


    # ff.write('\n')
    # ff.write("bad_L: {}, bad_A:{}, bad_B:{}".format(bad_l, bad_a, bad_b) + '\n')
    # ff.write("L A B all diff in  0.5: {}, all data size: {}".format(c, len(ks)) + '\n')


    if not green_blue:
        plt.title("green data: gammaed_rgb diff ok_ng case")
        for gamma_ed_rgb, diff_a in blue_bad_a_dict.items():
            gammed_rgb = [float(a) for a in gamma_ed_rgb.split(',')[:-1]]
            if diff_a[0] > 0.5:
                plt.plot(aa, gammed_rgb, color='black', label='diff ng')
            else:
                plt.plot(aa, gammed_rgb, color='pink')
        plt.legend()
        plt.show()


def overfiting(X, Y, index, green_blue):
    dfull = xgb.DMatrix(X, Y)

    param1 = json.load(open(r'./parameter1_{}_{}.json'.format(index, green_blue), 'r'))
    num_round = 200

    cvresult1 = xgb.cv(param1, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()



def show_train_test(train, test):
    aa = [i for i in range(3)]
    plt.subplot(121)
    for a in test:
        plt.plot(aa, a, color='blue')
    plt.grid()
    plt.subplot(122)
    for a in train:
        plt.plot(aa, a, color='pink')
    plt.show()


if __name__ == "__main__":

    # merge data1 and 2
    prepare_data()

    js_x = json.load(open(r'./all_data_rgb3.json', 'r'))
    js_y = json.load(open(r'./all_data_lab.json', 'r'))

    # green: 0, blue: 1
    green_blue = 1

    flags = ['x', 'y', 'z']
    txts = ["green", "blue"]
    ff = open(r'./bad_{}.txt'.format(txts[green_blue]), 'w')
    X_dict = dict()
    seeds = [11,22,33,44,55,66,77,88,99]
    for seed in seeds:
        for i in range(3):
            X, Y, rgb_ImgName, X_dict = load_data(js_x, js_y, i, green_blue, gammaed=True)
            assert X.shape[0] == Y.shape[0]

            X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=seed)
            # hyperparameter_searching(X, Y, i, green_blue)
            # overfiting(X, Y, i, green_blue)
            cross_val(X_train, y_train, X, X_test, i, green_blue, rgb_ImgName)
            # show train test的数据分布
            if i == 2:
                # 只show一次就可以
                show_train_test(X_train, X_test)
        # compare result
        check_lab_res(green_blue, js_x, js_y, ff, X_dict)


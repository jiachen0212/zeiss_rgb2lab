# coding=utf-8
'''
0812blue+0924blue =

'''

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb
import json
import warnings
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import os


def merge_data():
    js1_all = dict()
    js2_all = dict()
    js1 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812blue_lab.json', 'r'))
    js1_1 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924blue_lab.json', 'r'))

    js2 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812blue_rgb.json', 'r'))
    js2_1 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924blue_rgb.json', 'r'))


    js1_all.update(js1)
    js1_all.update(js1_1)
    js2_all.update(js2)
    js2_all.update(js2_1)

    data = json.dumps(js1_all)
    with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924_blue_lab.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(js2_all)
    with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924_blue_rgb.json', 'w') as js_file:
        js_file.write(data)
    # print(len(js2_all))

    assert len(js1_all) == len(js2_all)


def cross_val(tmp_dir, save_params_dir, X_train, y_train, X, X_test, index, rgb_ImgName):
    xyz_res = dict()
    single_xyz_res = os.path.join(tmp_dir, 'xyz_{}.json'.format(index))

    parameters = json.load(open(os.path.join(save_params_dir, 'parameter_green_{}.json'.format(index)), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, y_train)

    # test all data
    y_pred = xgb_model.predict(X)
    for ii, item in enumerate(y_pred):
        value = ''.join(str(a) for a in X[ii])
        info = rgb_ImgName[value]
        xyz_res[info] = str(item)

    data = json.dumps(xyz_res)
    with open(single_xyz_res, 'w') as js_file:
        js_file.write(data)



def report_best_scores(results, index, save_params_dir, n_top=3):
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
    with open(os.path.join(save_params_dir,  r'parameter_green_{}.json'.format(index)), 'w') as js_file:
        js_file.write(data)



def hyperparameter_searching(X, Y, index, save_params_dir):

    xgb_model = xgb.XGBRegressor()
    params = {
        "colsample_bytree": uniform(0.9, 0.1),
        "gamma": uniform(0, 0.),   # gamma越小, 模型越复杂..
        "learning_rate": uniform(0.01, 0.5),  # default 0.1
        "max_depth": randint(2, 10),  # default 3
        "n_estimators": randint(80, 150),  # default 100
        "subsample": uniform(0.6, 0.4)

    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, index, save_params_dir, 5)



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


def gamma(a):
    if a > 0.04045:
        a = np.power((a+0.055)/1.055, 2.4)
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



def show_b_gamma(org):
    aa = [0, 1, 2]
    plt.title(r"org rgb value")
    for ii, b1b2 in enumerate(org):
        plt.scatter(aa, b1b2, color='pink', s=2)
    plt.show()



def load_data(rgb, lab, index, gammaed=False):
    X_dict = dict()
    rgb_ImgName = dict()
    X , Y = [], []
    for k, v in rgb.items():
        r_, g_, b_ = [float(a) / 255 for a in v]
        if not gammaed:
            X.append([r_, g_, b_])
            v_ = lab2xyz(lab[k][0], lab[k][1], lab[k][2])
            Y.append(v_[index])
            rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k
            X_dict[k] = [r_, g_, b_]
        else:
            gamma_r_ = gamma(r_)
            gamma_g_ = gamma(g_)
            gamma_b_ = gamma(b_)
            X.append([gamma_r_, gamma_g_, gamma_b_])
            X_dict[k] = [gamma_r_, gamma_g_, gamma_b_]
            v_ = lab2xyz(lab[k][0], lab[k][1], lab[k][2])
            Y.append(v_[index])
            rgb_ImgName[''.join(str(a) for a in [gamma_r_, gamma_g_, gamma_b_])] = k

    # show_gammaed_rgb
    for tmp in X:
        plt.plot([0,1,2], tmp, color='pink')
    plt.show()

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, rgb_ImgName, X_dict


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


def check_lab_res(seed, tmp_dir, js_y, X_dict):

    aa = [i for i in range(3)]
    green_bad_a_dict = dict()

    x_pred = json.load(open(os.path.join(tmp_dir, 'xyz_0.json'), 'r'))
    y_pred = json.load(open(os.path.join(tmp_dir, 'xyz_1.json'), 'r'))
    z_pred = json.load(open(os.path.join(tmp_dir, 'xyz_2.json'), 'r'))

    c = 0
    # blue_diff = open(r'./green_diff.txt', 'w')
    for k, v in x_pred.items():
        real_l, real_a, real_b = js_y[k]
        pre_x, pre_y, pre_z = float(x_pred[k]), float(y_pred[k]), float(z_pred[k])
        pre_l, pre_a, pre_b = xyz2lab(pre_x, pre_y, pre_z)

        if abs(pre_l-real_l) <= 0.5 and abs(pre_a-real_a) <= 0.5 and abs(pre_b-real_b) <= 0.5:
            c += 1
        else:
            line = "data: {}, diff l: {}, diff a: {}, diff b: {}".format(str(int(k.split('_')[0])-50) + '_' + k.split('_')[1], (pre_l-real_l), (pre_a-real_a), (pre_b-real_b))
            # print(line)
            # blue_diff.write(line+'\n')

        green_bad_a_dict[''.join(str(a)+',' for a in X_dict[k])] = [abs(pre_a-real_a), k]

    print("seed: {}, L A B all diff in  0.5: {}, all data size: {}".format(seed, c, len(x_pred)))

    return c

def overfiting(X, Y, index, save_params_dir):
    dfull = xgb.DMatrix(X, Y)

    param1 = json.load(open(os.path.join(save_params_dir, r'parameter_green_{}.json'.format(index)), 'r'))
    num_round = 200

    cvresult1 = xgb.cv(param1, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # merge data1 and 0924
    merge_data()

    # all data
    # LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924_blue_lab.json', 'r'))
    # RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924_blue_rgb.json', 'r'))

    # 0812 blue
    # LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812blue_lab.json', 'r'))
    # RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812blue_rgb.json', 'r'))

    # 0924 blue
    # gammaed_b 要分区间, 然后提取lab相近, 但rgb差很多的异常样本.. 参考: 0924blue_split_gammaed_b
    LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924blue_lab.json', 'r'))
    RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924blue_rgb.json', 'r'))

    save_params_dir = r'D:\work\project\卡尔蔡司膜色缺陷\green_params_js'

    tmp_dir = r'D:\work\project\卡尔蔡司膜色缺陷\tmp_xyz_res_js'
    X_dict = dict()

    # 交叉验证
    seeds = [11*i for i in range(1, 20)]
    res = 0
    for seed in seeds:
        for i in range(3):
            X, Y, rgb_ImgName, X_dict = load_data(RGB, LAB, i, gammaed=True)
            X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=seed)
            # hyperparameter_searching(X, Y, i, save_params_dir)
            # overfiting(X, Y, i, save_params_dir)

            cross_val(tmp_dir, save_params_dir, X_train, y_train, X, X_test, i, rgb_ImgName)

        # compare results
        count = check_lab_res(seed, tmp_dir, LAB, X_dict)
        res += count

    print("交叉验证的acc: {}".format(res/(len(seeds)*len(X_dict))))



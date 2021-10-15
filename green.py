# coding=utf-8
'''
0812之前的全量green数据116 + 0924的78绿数据  all: 116+78 = 194

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
    js1 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_lab.json', 'r'))
    js1_1 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924_green_lab_slim.json', 'r'))

    js2 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_rgb.json', 'r'))
    js2_1 = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924_green_rgb_slim.json', 'r'))

    js1_all.update(js1)
    js1_all.update(js1_1)
    js2_all.update(js2)
    js2_all.update(js2_1)

    data = json.dumps(js1_all)
    with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924slim_green_lab.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(js2_all)
    with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924slim_green_rgb.json', 'w') as js_file:
        js_file.write(data)

    assert len(js1_all) == len(js2_all)


def cross_val(tmp_dir, save_params_dir, X_train, y_train, X, index, rgb_ImgName):
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



def cross_val_(tmp_dir, save_params_dir, X_train, y_train, X_test, index, rgb_ImgName):
    xyz_res = dict()
    single_xyz_res = os.path.join(tmp_dir, 'xyz_{}.json'.format(index))

    parameters = json.load(open(os.path.join(save_params_dir, 'parameter_green_{}.json'.format(index)), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X_train, y_train)

    # test all data
    y_pred = xgb_model.predict(X_test)
    for ii, item in enumerate(y_pred):
        value = ''.join(str(a) for a in X_test[ii])
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
            # print(r_, g_, b_)
            # print(gamma_r_, gamma_g_, gamma_b_)
            X.append([gamma_r_, gamma_g_, gamma_b_])
            X_dict[k] = [gamma_r_, gamma_g_, gamma_b_]
            v_ = lab2xyz(lab[k][0], lab[k][1], lab[k][2])
            Y.append(v_[index])
            rgb_ImgName[''.join(str(a) for a in [gamma_r_, gamma_g_, gamma_b_])] = k

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, rgb_ImgName, X_dict


def generate_test_data(rgb, lab, index, gammaed=False):
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
            print(line)
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


def split_0812_green(data1_lab, data1_rgb,  lab0812, rgb0812):
    # 0812有17条绿膜数据, split train and test train+data1, test做测试数据. 迁移性很差..
    ks = list(lab0812.keys())
    tmp1, tmp2 = dict(), dict()
    for i in range(15):
        tmp1[ks[i]] = lab0812[ks[i]]
        tmp2[ks[i]] = rgb0812[ks[i]]
    data1_lab.update(tmp1)
    data1_rgb.update(tmp2)

    test_lab, test_rgb = dict(), dict()
    for i in range(15, len(ks)):
        test_lab[ks[i]] = lab0812[ks[i]]
        test_rgb[ks[i]] = rgb0812[ks[i]]

    return data1_lab, data1_rgb, test_lab, test_rgb



def del_bad_rgb_lab_data(data_rgb, data_lab):
    tmp = ['25_14', '20_17', '18_7', '24_15', '25_13', '17_14', '24_16', '17_20', '17_3', '17_11', '17_6', '20_18', '17_1', '17_2', '24_10', '18_13', '17_15', '18_11', '17_18', '25_2', '24_1', '18_12', '18_1', '17_19', '17_8', '24_20', '25_4', '24_11', '16_7', '17_13', '17_12', '24_4']

    for k in tmp:
        del data_rgb[k]
        del data_lab[k]

    data = json.dumps(data_rgb)
    with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924_green_rgb_slim.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(data_lab)
    with open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924_green_lab_slim.json', 'w') as js_file:
        js_file.write(data)

    return data_rgb, data_lab



if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # merge data1 and 0924
    merge_data()

    # 0924
    LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924green_lab.json', 'r'))
    RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0924green_rgb.json', 'r'))
    # RGB, LAB = del_bad_rgb_lab_data(RGB, LAB)

    # all data
    # LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924slim_green_lab.json', 'r'))   # data1_0924_green_lab
    # RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_0924slim_green_rgb.json', 'r'))

    # data1
    # LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_lab.json', 'r'))
    # RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\data1_rgb.json', 'r'))


    # 0812 green data
    # LAB_ = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812green_lab.json', 'r'))
    # RGB_ = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\data\0812green_rgb.json', 'r'))
    # LAB, RGB, LAB_, RGB_ = split_0812_green(LAB, RGB, LAB_, RGB_)
    # assert len(LAB) == 116 + 15
    # assert len(LAB_) == 2

    save_params_dir = r'D:\work\project\卡尔蔡司膜色缺陷\green_params_js'

    tmp_dir = r'D:\work\project\卡尔蔡司膜色缺陷\tmp_xyz_res_js'
    X_dict = dict()
    X_dict_ = dict()

    # 交叉验证
    seeds = [11*i for i in range(1, 20)]
    res = 0
    for seed in seeds:
        for i in range(3):
            X, Y, rgb_ImgName, X_dict = load_data(RGB, LAB, i, gammaed=True)
            # X_, Y_, rgb_ImgName_, X_dict_ = generate_test_data(RGB_, LAB_, i, gammaed=True)
            # print(len(rgb_ImgName))
            X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=seed)
            # hyperparameter_searching(X, Y, i, save_params_dir)
            # overfiting(X, Y, i, save_params_dir)

            cross_val(tmp_dir, save_params_dir, X_train, y_train, X, i, rgb_ImgName)
            # cross_val_(tmp_dir, save_params_dir, X_train, y_train, X_, i, rgb_ImgName_)

        # compare results
        count = check_lab_res(seed, tmp_dir, LAB, X_dict)
        # count = check_lab_res(seed, tmp_dir, LAB_, X_dict_)
        res += count

    print("交叉验证的acc: {}".format(res/(len(seeds)*len(X_dict))))



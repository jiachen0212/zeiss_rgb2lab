# coding=utf-8
'''
sao操作: 0812+0924所有的蓝膜数据做参数搜索, 得到params. 然后仅用0924的蓝膜数据+params训xgboost

不ok...

'''
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import json
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as TTS

def prepare_data():
    js1_all = dict()
    js2_all = dict()
    js1 = json.load(open(r'./all_col3.json', 'r'))
    js1_1 = json.load(open(r'./all_col3_0821.json', 'r'))

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


def cross_val(X, Y, index, green_blue, rgb_ImgName):
    xyz_res = dict()
    single_xyz_res = r'./xyz_{}.json'.format(index)

    X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3, random_state=88)

    # hyperparameter_searching beat parameters
    parameters = json.load(open(r'./parameter_{}_{}.json'.format(index, green_blue), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:linear", **parameters)
    xgb_model.fit(X, Y)

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
    with open(r'./parameter_{}_{}.json'.format(index, green_blue), 'w') as js_file:
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
        "gamma": uniform(0, 0.),   # gamma越小, 模型越复杂..
        "learning_rate": uniform(0.01, 0.5),  # default 0.1
        "max_depth": randint(2, 10),  # default 3
        "n_estimators": randint(80, 150),  # default 100
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



def load_data(json_x, json_y, index, green_blue, gammaed=False):
    X_dict = dict()
    org_rgb = []
    gammed_rgb = []
    rgb_ImgName = dict()
    X , Y = [], []
    bad_g = 0
    for k, v in json_x.items():
        r_, g_, b_ = [float(a) / 255 for a in json_x[k]]
        if gammaed:
            org_rgb.append([r_, g_, b_])
            gamma_r_ = gamma(r_)
            gamma_g_ = gamma(g_)
            gamma_b_ = gamma(b_)
            X.append([gamma_r_, gamma_g_, gamma_b_])
            X_dict[k] = [gamma_r_, gamma_g_, gamma_b_]
            v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
            Y.append(v_[index])
            rgb_ImgName[''.join(str(a) for a in [gamma_r_, gamma_g_, gamma_b_])] = k
        else:
            # 只对r, g gamma
            g_ = gamma(g_)
            r_ = gamma(r_)
            if g_ < 0.4:
                X.append([r_, g_, b_])
                X_dict[k] = [r_, g_, b_]
                v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                Y.append(v_[index])
                rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k


    X = np.array(X)
    Y = np.array(Y)
    print("green value 异常的值: {}".format(bad_g))
    print(X.shape)
    print(len(rgb_ImgName))
    print(len(X_dict))

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



def split_blueand_green():
    # 蓝绿样本划分
    for k, v in js_x.items():
        v = [int(a) for a in v]
        dir_index = int(k.split('_')[0])
        # rgb 判断g值是否大于b值即可.
        if v[1] > v[2]:
            if dir_index > 21:
                print(k, '==')



import cv2
def imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def check_lab_res(green_blue, js_x, js_y, ff, X_dict):
    aa = [i for i in range(3)]

    blue_bad_a_dict = dict()
    x_pred = json.load(open(r'./xyz_0.json', 'r'))
    y_pred = json.load(open(r'./xyz_1.json', 'r'))
    z_pred = json.load(open(r'./xyz_2.json', 'r'))

    c = 0
    blue_diff = open(r'./blue_diff.txt', 'w')
    for k, v in x_pred.items():
        real_l, real_a, real_b = js_y[k]
        pre_x, pre_y, pre_z = float(x_pred[k]), float(y_pred[k]), float(z_pred[k])
        pre_l, pre_a, pre_b = xyz2lab(pre_x, pre_y, pre_z)

        if abs(pre_l-real_l) <= 0.5 and abs(pre_a-real_a) <= 0.5 and abs(pre_b-real_b) <= 0.5:
            c += 1
        else:
            line = "data: {}, diff l: {}, diff a: {}, diff b: {}".format(str(int(k.split('_')[0])-50) + '_' + k.split('_')[1], (pre_l-real_l), (pre_a-real_a), (pre_b-real_b))
            print(line)
            blue_diff.write(line+'\n')

        if green_blue:
            blue_bad_a_dict[''.join(str(a)+',' for a in X_dict[k])] = [abs(pre_a-real_a), k]


    bad_a = []
    ok_a = []
    tmp_save_dir = r'C:\Users\15974\Desktop\ycy'
    import cv2
    import os
    if green_blue:
        diff_b_g = []
        plt.title("gamma_ed_rgb diff ok_ng case")
        for gamma_ed_rgb, diff_a in blue_bad_a_dict.items():
            gammed_rgb = [float(a) for a in gamma_ed_rgb.split(',')[:-1]]
            im_name = diff_a[1].split('_')[1] + '.bmp'
            img_path = os.path.join(r'D:\work\project\卡尔蔡司AR镀膜\poc\20210924\20210924',
                                    str(int(diff_a[1].split('_')[0]) - 50), im_name)
            # img = imread(img_path)

            if diff_a[0] > 0.5:
                # print("bad a color: {}".format(js_x[diff_a[1]]))
                bad_a.append(diff_a[1])
                # tmp_path = os.path.join(tmp_save_dir, 'bad', str(int(diff_a[1].split('_')[0])-50))
                # if not os.path.exists(tmp_path):
                #     os.mkdir(tmp_path)
                # cv2.imwrite(os.path.join(tmp_path, im_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # print(js_x[diff_a[1]][2] - js_x[diff_a[1]][1])
                # if js_x[diff_a[1]][2] - js_x[diff_a[1]][1] < 100:
                #     plt.plot(aa, gammed_rgb, color='black')
                # else:
                #     plt.plot(aa, gammed_rgb, color='green')
                plt.plot(aa, gammed_rgb, color='black')
            else:
                # print("ok a color: {}".format(js_x[diff_a[1]]))
                # diff_b_g.append(js_x[diff_a[1]][2] - js_x[diff_a[1]][1])
                # tmp_path = os.path.join(tmp_save_dir, 'ok', str(int(diff_a[1].split('_')[0]) - 50))
                # if not os.path.exists(tmp_path):
                #     os.mkdir(tmp_path)
                # cv2.imwrite(os.path.join(tmp_path, im_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                plt.plot(aa, gammed_rgb, color='pink')
                ok_a.append(diff_a[1])
        # ok样本, b-g的值没有显著的范围特点..
        # print(min(diff_b_g), max(diff_b_g))
        plt.show()

    # print("bad a: {}".format(bad_a))
    # print("ok a: {}".format(ok_a))
    print("L A B all diff in  0.5: {}, all data size: {}".format(c, len(x_pred)))

def overfiting(X, Y, index, green_blue):
    dfull = xgb.DMatrix(X, Y)

    param1 = json.load(open(r'./parameter_{}_{}.json'.format(index, green_blue), 'r'))
    num_round = 200

    cvresult1 = xgb.cv(param1, dfull, num_round)

    fig, ax = plt.subplots(1, figsize=(15,8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    plt.show()


if __name__ == "__main__":

    js_x = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_color.json', 'r'))
    js_y = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\poc\blue_0926 backup\blue_lab.json', 'r'))

    print("all blue data: {}".format(len(js_y)))
    green_blue = 1

    flags = ['x', 'y', 'z']
    txts = ["green", "blue"]
    ff = open(r'./bad_{}.txt'.format(txts[green_blue]), 'w')
    X_dict = dict()
    for i in range(3):
        print("for {} value".format(flags[i]))
        X, Y, rgb_ImgName, X_dict = load_data(js_x, js_y, i, green_blue, gammaed=True)
        assert X.shape[0] == Y.shape[0]

        # use xgboost
        # hyperparameter_searching(X, Y, i, green_blue)
        overfiting(X, Y, i, green_blue)
        cross_val(X, Y, i, green_blue, rgb_ImgName)

    # compare result
    check_lab_res(green_blue, js_x, js_y, ff, X_dict)


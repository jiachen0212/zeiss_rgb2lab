# coding=utf-8
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import json
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as TTS
import os
import collections


def cross_val(tmp_dir, save_params_dir, X_train, y_train, X, index, rgb_ImgName, yc_x, yc_rgb_ImgName):
    xyz_res = dict()
    yc_xyz_res = dict()
    single_xyz_res = os.path.join(tmp_dir, 'xyz_{}.json'.format(index))
    yc_xyz = os.path.join(tmp_dir, 'yc_xyz_{}.json'.format(index))

    parameters = json.load(open(os.path.join(save_params_dir, 'parameter_green_{}.json'.format(index)), 'r'))

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", **parameters)
    xgb_model.fit(X_train, y_train)

    # test train data
    y_pred = xgb_model.predict(X)
    for ii, item in enumerate(y_pred):
        value = ''.join(str(np.round(a, 3)) for a in X[ii])
        info = rgb_ImgName[value]
        xyz_res[info] = str(item)
    data = json.dumps(xyz_res)
    with open(single_xyz_res, 'w') as js_file:
        js_file.write(data)

    y_pred_yc = xgb_model.predict(yc_x)
    for ii, item in enumerate(y_pred_yc):
        value = ''.join(str(np.round(a, 3)) for a in yc_x[ii])
        info = yc_rgb_ImgName[value]
        yc_xyz_res[info] = str(item)
    data = json.dumps(yc_xyz_res)
    with open(yc_xyz, 'w') as js_file:
        js_file.write(data)



def cross_val_test(seed, tmp_dir, save_params_dir, X, Y, index, test_x, test_rgb_ImgName):
    test_xyz_res = dict()
    test_xyz = os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, index))
    parameters = json.load(open(os.path.join(save_params_dir, 'parameter_green_{}.json'.format(index)), 'r'))
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", **parameters)
    xgb_model.fit(X, Y)

    y_pred = xgb_model.predict(test_x)
    for ii, item in enumerate(y_pred):
        value = ''.join(str(np.round(a, 3)) for a in test_x[ii])
        info = test_rgb_ImgName[value]
        test_xyz_res[info] = str(item)
    data = json.dumps(test_xyz_res)
    with open(test_xyz, 'w') as js_file:
        js_file.write(data)



def report_best_scores(results, index, save_params_dir, n_top=3):
    parameter = dict()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            parameter = results['params'][candidate]

    data = json.dumps(parameter)
    with open(os.path.join(save_params_dir,  r'parameter_green_{}.json'.format(index)), 'w') as js_file:
        js_file.write(data)


def hyperparameter_searching_1(X, Y, index, save_params_dir):

    xgb_model = xgb.XGBRegressor()
    params = {
        "learning_rate": uniform(0.05, 0.5),
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=6, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, index, save_params_dir, 5)


def hyperparameter_searching(X, Y, index, save_params_dir):

    xgb_model = xgb.XGBRegressor()
    params = {
        "learning_rate": uniform(0.05, 0.2),
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, index, save_params_dir, 5)


def hyperparameter_searching_2(X, Y, index, save_params_dir):

    xgb_model = xgb.XGBRegressor(colsample_bytree=0.924, gamma=0, max_depth=7, n_estimators=131, subsample=0.745)
    params = {
        "learning_rate": uniform(0.05, 0.1),
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(X, Y)

    report_best_scores(search.cv_results_, index, save_params_dir, 5)


def hyperparameter_searching_0(X, Y, index, save_params_dir):

    xgb_model = xgb.XGBRegressor(colsample_bytree=0.945, gamma=0, max_depth=5, lr=0.15, subsample=0.9)
    params = {
        "n_estimators": randint(80, 150),
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
    ks = []
    bad_green_data = []
    gammaed_r, gammaed_g, gammaed_b = [], [], []
    R,G,B = [], [], []
    for k, v in rgb.items():
        [r_, g_, b_] = [float(v[i]) / 255 for i in range(3)]
        if not gammaed:
            X.append([r_, g_, b_])
            v_ = lab2xyz(lab[k][0], lab[k][1], lab[k][2])
            Y.append(v_[index])
            rgb_ImgName[''.join(str(a) for a in [r_, g_, b_])] = k
            X_dict[k] = [r_, g_, b_]
        else:
            R.append(r_)
            G.append(g_)
            B.append(b_)
            gamma_r_ = gamma(r_)
            gamma_g_ = gamma(g_)
            gamma_b_ = gamma(b_)
            k_gb = gamma_g_ / gamma_b_
            if k_gb < 1:
                bad_green_data.append(k)
            gammaed_r.append(gamma_r_)
            gammaed_g.append(gamma_g_)
            gammaed_b.append(gamma_b_)
            x = [gamma_r_, gamma_g_, gamma_b_] #  + [np.exp(r_), np.exp(g_), np.exp(b_)]
            X.append(x)
            X_dict[k] = x
            v_ = lab2xyz(lab[k][0], lab[k][1], lab[k][2])
            Y.append(v_[index])
            ks.append(k)
            # rgb_ImgName[''.join(str(a) for a in x)] = k

    for ind in range(len(gammaed_g)):
        plt.plot([i for i in range(3)], [gammaed_r[ind], gammaed_g[ind], gammaed_b[ind]], color='pink')
    # plt.grid()
    # plt.show()

    # normalize
    mean_ = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)
    ff = open('./mean_std.txt', 'w')
    for m in mean_:
        ff.write(str(m)+',')
    ff.write('\n')
    for m in std_:
        ff.write(str(m)+',')
    ff.write('\n')

    X = [[(x[i]-mean_[i])/std_[i] for i in range(len(std_))] for x in X]
    for ind, x in enumerate(X):
        rgb_ImgName[''.join(str(np.round(a, 3)) for a in x)] = ks[ind]

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, rgb_ImgName, X_dict, bad_green_data


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
        print("1")
        fy = np.power(y, 1/3)
    else:
        print("2")
        fy = 7.787 * y + 16 / 116.0
    if x > 0.008856:
        print("3")
        fx = np.power(x, 1/3)
    else:
        print("4")
        fx = 7.787 * x + 16 / 116.0
    if z > 0.008856:
        print("5")
        fz = np.power(z, 1/3)
    else:
        print("6")
        fz = 7.787 * z + 16 / 116.0
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return [l, a, b]



def check_lab_res(res_txt, seed, tmp_dir, js_y, X_dict):
    L = [9, 14]
    A = [-24, -15]
    B = [-2, 10]
    green_bad_a_dict = dict()

    x_pred = json.load(open(os.path.join(tmp_dir, 'xyz_0.json'), 'r'))
    y_pred = json.load(open(os.path.join(tmp_dir, 'xyz_1.json'), 'r'))
    z_pred = json.load(open(os.path.join(tmp_dir, 'xyz_2.json'), 'r'))

    # yc lab data test
    yc_x_pred = json.load(open(os.path.join(tmp_dir, 'yc_xyz_0.json'), 'r'))
    yc_y_pred = json.load(open(os.path.join(tmp_dir, 'yc_xyz_1.json'), 'r'))
    yc_z_pred = json.load(open(os.path.join(tmp_dir, 'yc_xyz_2.json'), 'r'))

    all_yc = len(yc_x_pred)

    for k, v in yc_x_pred.items():
        pre_x, pre_y, pre_z = float(v), float(yc_y_pred[k]), float(yc_z_pred[k])
        pre_l, pre_a, pre_b = xyz2lab(pre_x, pre_y, pre_z)
        if (L[0] <= pre_l <= L[1]) and (A[0] <= pre_a <= A[1]) and (B[0] <= pre_b <= B[1]):
            all_yc -= 1
    # print("pred yc: {}".format(all_yc))
    c = 0
    l,a,b = 0,0,0
    ngs = []
    preds_lab = dict()
    for k, v in x_pred.items():
        real_l, real_a, real_b = js_y[k]
        pre_x, pre_y, pre_z = float(x_pred[k]), float(y_pred[k]), float(z_pred[k])
        pre_l, pre_a, pre_b = xyz2lab(pre_x, pre_y, pre_z)
        preds_lab[k] = [pre_l, pre_a, pre_b]

        if abs(pre_l-real_l) <= 0.5 and abs(pre_a-real_a) <= 0.5 and abs(pre_b-real_b) <= 0.5:
            c += 1
        else:
            line = "data: {}, diff l: {}, diff a: {}, diff b: {}".format(str(int(k.split('_')[0])) + '_' + k.split('_')[1], (pre_l-real_l), (pre_a-real_a), (pre_b-real_b))
            print(line)
            res_txt.write(line + '\n')
            ngs.append(k)
        if abs(pre_l-real_l) > 0.5:
            l += 1
        if abs(pre_a-real_a) > 0.5:
            a += 1
        if abs(pre_b-real_b) > 0.5:
            b += 1

        green_bad_a_dict[''.join(str(a)+',' for a in X_dict[k])] = [abs(pre_a-real_a), k]

    print("seed: {}, L A B all diff in  0.5: {}, all data size: {}".format(seed, c, len(x_pred)))
    print("bad l: {}, bad a: {}, bad b: {}".format(l,a,b))
    print(ngs)
    res_txt.write('\n')
    res_txt.write('\n')


    return c, ngs, preds_lab, all_yc


def get_test_lab(tmp_dir):
    L = [9.5, 14.5]
    A = [-24, -15]
    B = [-2, 10]

    x_pred = json.load(open(os.path.join(tmp_dir, '0_test_xyz_0.json'), 'r'))
    y_pred = json.load(open(os.path.join(tmp_dir, '0_test_xyz_1.json'), 'r'))
    z_pred = json.load(open(os.path.join(tmp_dir, '0_test_xyz_2.json'), 'r'))

    pred_lab = dict()
    for k, v in x_pred.items():
        pre_x, pre_y, pre_z = float(v), float(y_pred[k]), float(z_pred[k])
        pre_l, pre_a, pre_b = xyz2lab(pre_x, pre_y, pre_z)
        pre_l = np.round(pre_l, 2)
        pre_a = np.round(pre_a, 2)
        pre_b = np.round(pre_b, 2)
        if (L[0] <= pre_l <= L[1]) and (A[0] <= pre_a <= A[1]) and (B[0] <= pre_b <= B[1]):
            pred_lab[k] = [pre_l, pre_a, pre_b, '正常']
        else:
            pred_lab[k] = [pre_l, pre_a, pre_b, '异常']

    return pred_lab

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


def show_result(all_ngs, RGB):
    ff = open('./ycs.txt', 'r').readlines()[0]
    ycs = ff.split(',')[:-1]
    tmp = collections.Counter(all_ngs)
    top_ngs = sorted(tmp.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
    max_ng_time = top_ngs[0][1]
    tmp = []
    print(top_ngs)
    for ng in top_ngs:
        if ng[1] >= max_ng_time:
            tmp.append(ng[0])
    ts = []
    for t in tmp:
        if t in ycs:
            ts.append(t)
    print("高频出错且是lab异常的样本: {}".format(ts))

    # green_test = json.load(open(r'./1209_test_rgb.json', 'r'))
    # for k, v in green_test.items():
    #     plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color='red')

    color_ind = 0
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    for k, v in RGB.items():
        if k not in tmp:
            plt.plot([0,1,2], [gamma(float(a)/255) for a in v], color='pink')
        else:
            print(k, colors[color_ind])
            plt.plot([0, 1, 2], [gamma(float(a) / 255) for a in v], color=colors[color_ind])
            color_ind += 1
    plt.show()


def get_test_data(test_rgb):
    test_rgb_ImgName = dict()

    X = []
    ks = []
    ff = open('./mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:3]]
    std_ = [float(a) for a in ff[1].split(',')[:3]]
    for k, v in test_rgb.items():
        x = [gamma(float(a)/255) for a in v]
        x = [(x[i] - mean_[i]) / std_[i] for i in range(len(std_))]
        ks.append(k)
        X.append(x)
    X = np.array(X)

    for ind, x in enumerate(X):
        test_rgb_ImgName[''.join(str(np.round(a, 3)) for a in x)] = ks[ind]

    return X, test_rgb_ImgName


def get_yc_data():
    yc_rgb_ImgName = dict()
    test_rgb = json.load(open(r'./1209_yc_rgb.json', 'r'))

    X = []
    ks = []
    ff = open('./mean_std.txt', 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:3]]
    std_ = [float(a) for a in ff[1].split(',')[:3]]
    for k, v in test_rgb.items():
        x = [gamma(float(a)/255) for a in v]
        x = [(x[i] - mean_[i]) / std_[i] for i in range(len(std_))]
        ks.append(k)
        X.append(x)
    X = np.array(X)

    for ind, x in enumerate(X):
        yc_rgb_ImgName[''.join(str(np.round(a, 3)) for a in x)] = ks[ind]

    return X, yc_rgb_ImgName


def TestReslut(seeds, test_gt):
    seeds.remove(11)
    # 输出test data pred lab_value
    res = dict()
    for seed in seeds:
        js_ = json.load(open('./seed_{}_test_lab.json'.format(seed), 'r'))
        for k, v in js_.items():
            res[k] = [0,0,0]
        break

    # 均值一下10次交叉验证模型的lab预测值
    for seed in seeds:
        js_ = json.load(open('./seed_{}_test_lab.json'.format(seed), 'r'))
        for k, v in js_.items():
            res[k] = [res[k][i] + v[i] for i in range(3)]

    tmp = {'5_10': [11.28, -18.56, -0.65], '5_9': [11.33, -18.78, 0.07], '5_13': [11.33, -18.77, 0.23]}
    for k, v in res.items():
        v = [np.round(a/len(seeds), 2) for a in v]
        if k in ['5_9', '5_13', '5_10']:
            print("img: {}, pred: {}, real: {}".format(k, v, tmp[k]))
        res[k] = v

    data = json.dumps(res)
    with open('./test_data_lab.json', 'w') as js_file:
        js_file.write(data)

    # for seed in seeds:
    #     js_ = json.load(open('./seed_{}_test_lab.json'.format(seed), 'r'))
    #     for k, v in js_.items():
    #         diff_k = [abs(v[i] - res[k][i]) for i in range(3)]
    #         if (diff_k[0] >= 0.5) or (diff_k[1] >= 0.5) or (diff_k[2] >= 0.5):
    #             print(seed, k, diff_k)

    for k, v in test_gt.items():
        pred = [float(a) for a in res[k]]
        diff = [np.round(abs(v[i] - pred[i]), 3) for i in range(3)]
        for di in diff:
            if di >= 0.5:
                print("img: {}, diff: {}".format(k, diff))
    for k in res:
        try:
            test_gt[k]
        except:
            print(k, res[k])
            continue

    # read_to_csv
    # LAB合规范围
    L = [9, 14]
    A = [-24, -15]
    B = [-2, 10]
    import pandas as pd
    df = pd.DataFrame()
    ks, Ls, As, Bs, ook = [], [], [], [], []
    for k, v in res.items():
        ks.append(k)
        l, a, b = v[0], v[1], v[2]
        Ls.append(l)
        As.append(a)
        Bs.append(b)
        if (L[0] <= l <= L[1]) and (A[0] <= a <= A[1]) and (B[0] <= b <= B[1]):
            ook.append(r"在范围内")
        else:
            ook.append(r"不在范围内")
    df['img_name'] = ks
    df['pred_L'] = Ls
    df['pred_A'] = As
    df['pred_B'] = Bs
    df['LAB是否在范围内'] = ook
    df.to_csv(r'./green_test_lab.csv')


def get_gt_lab():
    import xlrd
    test_gt = dict()
    test_gt_csv = r'./params_js_0.92/test_data_gt.xlsx'
    wb = xlrd.open_workbook(test_gt_csv)
    data = wb.sheet_by_name(r'Sheet1')
    rows = data.nrows
    for i in range(1, rows):
        im_name = data.cell(i, 0).value
        l, a, b = data.cell(i, 1).value, data.cell(i, 2).value, data.cell(i, 3).value
        test_gt[im_name] = [l, a, b]

    return test_gt



def seed_pred_result(seeds):
    # 删除和mean之后结果差异很大的seed_res
    test_lab = dict()
    pred_test_x = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(0, 0)), 'r'))
    for k in pred_test_x:
        test_lab[k] = [0, 0, 0]
    for seed in seeds:
        pred_test_lab = dict()
        pred_test_x = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, 0)), 'r'))
        pred_test_y = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, 1)), 'r'))
        pred_test_z = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, 2)), 'r'))
        for k, v in pred_test_x.items():
            pre_x, pre_y, pre_z = float(v), float(pred_test_y[k]), float(pred_test_z[k])
            predl, preda, predb = xyz2lab(pre_x, pre_y, pre_z)
            pred_test_lab[k] = [predl, preda, predb]
            test_lab[k][0] += predl
            test_lab[k][1] += preda
            test_lab[k][2] += predb
        data = json.dumps(pred_test_lab)
        with open(r'./seed_{}_pred_test_lab.json'.format(seed), 'w') as js_file:
            js_file.write(data)
    test_ks = list(test_lab.keys())
    for k in test_ks:
        test_lab[k] = [a / len(seeds) for a in test_lab[k]]
    data = json.dumps(test_lab)
    with open(r'./mean_pred_test_lab.json', 'w') as js_file:
        js_file.write(data)

    bad_seeds = []
    for seed in seeds:
        pred_res = json.load(open(r'./seed_{}_pred_test_lab.json'.format(seed), 'r'))
        for k, v in pred_res.items():
            if (abs(v[0] - test_lab[k][0]) >= 0.5) or (abs(v[1] - test_lab[k][1]) >= 0.5) or (
                    abs(v[2] - test_lab[k][2]) >= 0.5):
                bad_seeds.append(seed)
    distribute = collections.Counter(bad_seeds)
    sored = sorted(distribute.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
    need_remove = [sored[0][0], sored[1][0], sored[2][0], sored[3][0]]
    print("need_remove: ", need_remove)
    for rem in need_remove:
        seeds.remove(rem)

    test_lab = dict()
    pred_test_x = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(0, 0)), 'r'))
    for k in pred_test_x:
        test_lab[k] = [0, 0, 0]
    for seed in seeds:
        pred_test_lab = dict()
        pred_test_x = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, 0)), 'r'))
        pred_test_y = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, 1)), 'r'))
        pred_test_z = json.load(open(os.path.join(tmp_dir, '{}_test_xyz_{}.json'.format(seed, 2)), 'r'))
        for k, v in pred_test_x.items():
            pre_x, pre_y, pre_z = float(v), float(pred_test_y[k]), float(pred_test_z[k])
            predl, preda, predb = xyz2lab(pre_x, pre_y, pre_z)
            pred_test_lab[k] = [predl, preda, predb]
            test_lab[k][0] += predl
            test_lab[k][1] += preda
            test_lab[k][2] += predb
        data = json.dumps(pred_test_lab)
        with open(r'./seed_{}_pred_test_lab.json'.format(seed), 'w') as js_file:
            js_file.write(data)
    test_ks = list(test_lab.keys())
    for k in test_ks:
        test_lab[k] = [a / len(seeds) for a in test_lab[k]]
    data = json.dumps(test_lab)
    with open(r'./0119_mean_pred_test_lab.json', 'w') as js_file:
        js_file.write(data)

    return test_lab


def train_data_all_in_model(RGB, LAB, test_x, test_rgb_ImgName):
    seed = 0
    for i in range(3):
        X, Y, rgb_ImgName, X_dict, bad_green_data = load_data(RGB, LAB, i, gammaed=True)
        cross_val_test(seed, tmp_dir, save_params_dir, X, Y, i, test_x, test_rgb_ImgName)
    test_lab = get_test_lab(tmp_dir)

    return test_lab

    # ims, Ls, As, Bs = [], [], [], []
    # LAB_range = []
    # for k, v in test_lab.items():
    #     ims.append(k)
    #     Ls.append(v[0])
    #     As.append(v[1])
    #     Bs.append(v[2])
    #     LAB_range.append(v[3])
    # ycy_df = pd.DataFrame()
    # ycy_df["id"] = ims
    # ycy_df["L"] = Ls
    # ycy_df["A"] = As
    # ycy_df["B"] = Bs
    # # ycy_df["LAB范围"] = LAB_range
    # ycy_df.to_csv('./1209_pred_lab.csv')

    # import xlrd
    # green_gt_dict = dict()
    # wb = xlrd.open_workbook(r'D:\work\project\卡尔蔡司膜色缺陷\1209\1216zeiss对齐材料\green_gt.xlsx')
    # data = wb.sheet_by_name(r'Sheet1')
    # rows = data.nrows
    # for i in range(1, rows):
    #     line = data.row_values(i)
    #     green_gt_dict[line[0]] = line[1:]
    # data = json.dumps(green_gt_dict)
    # with open(r'./1209_test_lab_gt.json', 'w') as js_file:
    #     js_file.write(data)


if __name__ == "__main__":

    LAB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\0119_train_lab.json', 'r'))
    RGB = json.load(open(r'D:\work\project\卡尔蔡司膜色缺陷\22.01.7data\0107all_erode_dilate.json', 'r'))

    del LAB["1_1"]
    del RGB["1_1"]
    del LAB["3_1"]
    del RGB["3_1"]

    test_rgb = dict()
    for k in list(RGB.keys()):
        if k not in LAB:
            test_rgb[k] = RGB[k]
            del RGB[k]
    print("test size {}".format(len(test_rgb)))

    save_params_dir = r'D:\work\project\卡尔蔡司膜色缺陷\1209\params_js_0.92\params_js_0.92'
    if not os.path.exists(save_params_dir):
        os.mkdir(save_params_dir)

    tmp_dir = r'./tmp_xyz_res_js'
    X_dict = dict()
    test_x, test_rgb_ImgName = get_test_data(test_rgb)
    yc_x, yc_rgb_ImgName = get_yc_data()

    all_ngs = []
    res_txt = open(r'./ycy.txt', 'w')
    seeds = [i * 11 for i in range(10)]
    res = 0
    for seed in seeds:
        for i in range(3):
            X, Y, rgb_ImgName, X_dict, bad_green_data = load_data(RGB, LAB, i, gammaed=True)
            X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=seed)
            cross_val(tmp_dir, save_params_dir, X_train, y_train, X, i, rgb_ImgName, yc_x, yc_rgb_ImgName)
            cross_val_test(seed, tmp_dir, save_params_dir, X_train, y_train, i, test_x, test_rgb_ImgName)
        count, ngs, preds_lab, pre_yc_count = check_lab_res(res_txt, seed, tmp_dir, LAB, X_dict)

        all_ngs.extend(ngs)
        res += count
    print("交叉验证的acc: {}".format(res/(len(seeds)*len(X_dict))))
    show_result(all_ngs, RGB)

    res = dict()
    res1 = seed_pred_result(seeds)
    res2 = train_data_all_in_model(RGB, LAB, test_x, test_rgb_ImgName)
    for k in res1:
        res[k] = [np.round((res1[k][r] + res2[k][r])/2, 2) for r in range(3)]
    print(res)
    test_gt = json.load(open(r'./0107_test_gt_lab1.json', 'r'))
    for k in res:
        abs_diff = [abs(np.round((res[k][r] - test_gt[k][r]), 2)) for r in range(3)]
        if abs_diff[0] >= 0.5 or abs_diff[1] >= 0.6 or abs_diff[2] >= 0.5:
            print(k, abs_diff)


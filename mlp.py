# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from data_load import DataLoader
import torch.optim as optimizers
from torch.autograd import Variable
import matplotlib.pyplot as plt
from util import calculate_Lab

np.random.seed(369)
torch.manual_seed(957)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.a3 = nn.ReLU()
        self.layers = [self.l1, self.a1, self.l2, self.a2, self.l3, self.a3]

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)

        return x


def read_data():
    js1_all = dict()
    js2_all = dict()
    js1 = json.load(open(r'./all_col6.json', 'r'))
    js2 = json.load(open(r'./all_lab.json', 'r'))
    js1_1 = json.load(open(r'./all_col6_0817.json', 'r'))
    print(len(js1_1))
    js2_1 = json.load(open(r'./all_lab_0817.json', 'r'))
    js1_all.update(js1)
    js1_all.update(js1_1)
    js2_all.update(js2)
    js2_all.update(js2_1)
    assert len(js1_all) == len(js2_all)
    print(len(js2_all))

    X, Y = [], []
    for k, v in js1_all.items():
        col3 = [int(a) for a in v]
        lab = js2_all[k]
        X.append(col3)
        # 拟合lab曲线
        # Y.append(lab)
        L, A, B = calculate_Lab(lab)
        Y.append([L,A,B])
    X = np.array(X)
    Y = np.array(Y)
    mean_ = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)
    # print(X)
    # normalize
    for i in range(X.shape[0]):
        X[i] = [(X[i][j] - mean_[j]) / std_[j] for j in range(X.shape[1])]
    # print(X)
    return X, Y


def compute_loss(t, y):
    return nn.MSELoss()(y, t)


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('poc')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


def show_y_pred(y, gt_y=None, epo=None, flag='val'):
    sample_num, dims = y.shape
    plt.title('{} epoch {} lab_curve'.format(flag, epo + 1))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        if i == 0:
            plt.plot(x, single_gt_y, color='pink', label='real')
            plt.plot(x, single_y, color='black', label='model result')
        else:
            plt.plot(x, single_gt_y, color='pink')
            plt.plot(x, single_y, color='black')
    plt.legend()
    plt.show()



def show_y_pred_single(y, gt_y=None, epo=None, flag='val'):
    sample_num, dims = y.shape
    plt.title('{} epoch {} lab_curve'.format(flag, epo + 1))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        # if single_gt_y[14] >= 4:
        #     plt.plot(x, single_gt_y, color='pink', label='real')
        #     plt.plot(x, single_y, color='black', label='model result')
        # elif single_gt_y[74] > 9:
        #     plt.plot(x, single_gt_y, color='red')
        #     plt.plot(x, single_y, color='green')
        # elif single_gt_y[24] > 2.82:
        #     plt.plot(x, single_gt_y, color='green')
        #     plt.plot(x, single_y, color='lightsalmon')
        if single_gt_y[14] >= 9.5:
            plt.plot(x, single_gt_y, color='pink', label='real')
            plt.plot(x, single_y, color='black', label='model result')
        # elif single_gt_y[74] < 4:
        #     plt.plot(x, single_gt_y, color='red')
        #     plt.plot(x, single_y, color='green')

    plt.legend()
    plt.show()


if __name__ == '__main__':

    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]

    # model = MLP(6, 80, 81).to('cpu')
    model = MLP(6, 200, 3).to('cpu')
    print(model)
    X, Y = read_data()
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.28, random_state=3)
    print("train size: {}, validation size: {}".format(train_x.shape[0], test_x.shape[0]))
    # train_dataloader = DataLoader((train_x, train_y), batch_size=train_x.shape[0], batch_first=False, device='cpu')
    train_dataloader = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device='cpu')
    val_dataloader = DataLoader((test_x, test_y), batch_size=test_x.shape[0], batch_first=False, device='cpu')
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)

    loss_list = []
    res = 0
    diff_value = 0
    epochs = 10000
    xx = [380+i*5 for i in range(81)]
    for epoch in range(epochs):
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data, requires_grad=False)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            print("real lab value: {},\tpred lab value: {}".format(label[0], score[0]))
            loss = compute_loss(score, target)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # print("epoch: {}, loss: {}".format(epoch, loss.item()))
            # if epoch == 1199:
            #     out_lab = score.detach().numpy()
            #     print(out_lab.shape)
                # show_y_pred_single(out_lab, label, epoch, flag='train')

            # if (epoch+1) % 100 == 0:
            #     for ii, (data, label) in enumerate(val_dataloader):
            #         model.eval()
            #         score = model(data)
            #         print(epoch, compute_loss(score, label).item())
            #         # show_y_pred(score.detach().numpy(), label, epoch, flag='val')
            #         # show_y_pred_single(score.detach().numpy(), label, epoch, flag='val')

            if epoch == epochs - 1:
                out_lab = score.detach().numpy()
                print(out_lab)
                # print(label)

'''
            if epoch == epochs-1:
                out_lab = score.detach().numpy()
                for index, curve in enumerate(out_lab):
                    l, a, b = calculate_Lab(curve)
                    real_curve = label.detach().numpy()[index]
                    gt_l, gt_a, gt_b = calculate_Lab(real_curve)
                    if abs(l-gt_l) <=0.5 and abs(l-gt_l) <=0.5 and abs(l-gt_l) <=0.5:
                        res += 1
                    elif abs(l-gt_l) > 5 or abs(l-gt_l) > 5 or abs(l-gt_l) > 5:
                        # diff_value += abs(l-gt_l) + abs(l-gt_l) + abs(l-gt_l)
                        plt.plot(xx, curve, color='black')
                        plt.plot(xx, real_curve, color='pink')
                        plt.plot(xx, best, color='red')
                        # plt.show()
                # plt.show()
                    else:
                        diff_value += abs(l - gt_l) + abs(l - gt_l) + abs(l - gt_l)
            
    # print(diff_value/(229-182))
    # plot_loss(loss_list)
    # print("L, A, B 三个值 diff [-.5, 0.5] 范围内个数: {}, 总数据量: {}, 精度为: {}".format(res, X.shape[0], res/float(X.shape[0])))
'''

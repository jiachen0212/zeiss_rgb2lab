import xlrd
import numpy as np

file = '/Users/chenjia/Downloads/Learning/SmartMore/2022/膜色缺陷/20220412镜片数据new.xlsx'
data = xlrd.open_workbook(file)
data= data.sheet_by_name('new')

me_x, me_y, me_z = data.col_values(3)[1:], data.col_values(4)[1:], data.col_values(5)[1:]
X, Y, Z = data.col_values(18)[1:], data.col_values(19)[1:], data.col_values(20)[1:]
L, A, B = data.col_values(21)[1:], data.col_values(22)[1:], data.col_values(23)[1:]

def fun(x):
    if x > 0.008856:
        x = np.power(x, 1/3)
    else:
        x = 7.787 * x + 16 / 116.0
    return x


def xyz2lab(x, y, z):
    x /= 94.81211415  # x /= Xn
    y /= 100
    z /= 107.3369399
    fy = fun(y)
    fx = fun(x)
    fz = fun(z)
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return [l, a, b]


deta_txt = open('./detaxyz.txt', 'w')
for i in range(len(me_x)):
    deta_x, deta_y, deta_z = me_x[i]-X[i], me_y[i]-Y[i], me_z[i]-Z[i]
    line = "deta_x: {}, deta_y: {}, deta_z: {}\n".format(np.round(deta_x, 4), np.round(deta_y, 4), np.round(deta_z, 4))
    # print("deta_x: {}, deta_y: {}, deta_z: {}".format(np.round(deta_x, 4), np.round(deta_y, 4), np.round(deta_z, 4)))
    deta_txt.write(line)


for i in range(len(me_x)):
    me_l, me_a, me_b = xyz2lab(me_x[i], me_y[i], me_z[i])
    deta_l, deta_a, deta_b = me_l-L[i], me_a-A[i], me_b-B[i]
    print("deta_l: {}, deta_a: {}, deta_b: {}".format(np.round(deta_l, 4), np.round(deta_a, 4), np.round(deta_b, 4)))
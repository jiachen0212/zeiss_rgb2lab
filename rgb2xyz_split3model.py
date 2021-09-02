def load_data(json_x, json_y, index, green_blue):
    rgb_ImgName = dict()
    X , Y = [], []
    for k, v in json_x.items():
        dir_index = int(k.split('_')[0])
        if not green_blue:
            if dir_index <= 21 or k in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                r_, g_, b_ = [float(a)/255 for a in json_x[k]]
                X.append([r_, g_, b_])
                v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                Y.append(v_[index])
                # print([r_, g_, b_])
                rgb_ImgName[''.join(str(int(a)/255) for a in json_x[k]) + str(v_[index])] = k
        else:
            if dir_index > 21 and k not in ["23_1", "23_2", "23_3", "23_4", "23_5"]:
                r_, g_, b_ = [float(a) / 255 for a in json_x[k]]
                X.append([r_, g_, b_])
                v_ = lab2xyz(json_y[k][0], json_y[k][1], json_y[k][2])
                Y.append(v_[index])
                rgb_ImgName[''.join(str(int(a)/255) for a in json_x[k]) + str(v_[index])] = k

import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from scipy import spatial
import os
import json


def sdk_pre(img_t):

    img =cv2.resize(img_t,(1024, 1024),interpolation=cv2.INTER_LINEAR)[np.newaxis,:,:,:]
    img = np.array(img,dtype=np.float32)
    img -=np.float32([123.675, 116.28, 103.53])
    img /= np.float32([58.395, 57.12, 57.375])
    img = np.transpose(img,[0,3,1,2])
    return img


def check_connect_comp(img,label_index):
    mask = np.array(img==label_index,np.uint8)
    num,label = cv2.connectedComponents(mask,8)
    return mask,num,label


def sdk_post(onnx_predict, predict, Confidence=0.0, Threshold=[0,0,0]):
    '''
    Args:
        Confidence: 缺陷的置信度，低于置信度的缺陷将会被过滤，默认0.8
        Threshold: [背景，铁屑，铁丝]的最小像素数量，少于最少像素的mask将会被过滤，默认[0,0,0]
    '''
    points = []
    num_class = predict.shape[1] 
    map_ = np.argmax(onnx_predict[0],axis=1)
    # print(f'pixel_classes: {np.unique(map_)}')
    mask_map = np.max(predict[0,:,:,:],axis=0)
    mask_ = map_[0,:,:]
    temo_predict=np.zeros(mask_.shape)
    score_print = np.zeros(mask_.shape)
    for i in range(num_class):
        if i == 0:
            continue
        else:
            _,num,label = check_connect_comp(mask_,i)
            print(num, '-=-=-=')
            for j in range(num):
                if j==0:
                    continue
                else:
                    temp=np.array(label==j,np.uint8)
                    score_temp = temp * mask_map
                    locate = np.where(temp>0)
                    number_thre = len(locate[0])
                    score_j = np.sum(score_temp)/number_thre
                    if number_thre > Threshold[i] and score_j > Confidence:
                        contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnt = contours[0]
                    
                        cnt = cnt.reshape(cnt.shape[0],2)
                        # print(cnt.shape)
                        if cnt.shape[0] < 3:
                            continue
                        print(f'classes: {i}, nums: {number_thre}')
                        candidates = cnt[spatial.ConvexHull(cnt).vertices]
                        dist_mat = spatial.distance_matrix(candidates,candidates)
                        i_, j_ = np.unravel_index(dist_mat.argmax(),dist_mat.shape)
                        
                        temo_predict += temp*i
                        points.append([candidates[i_], candidates[j_]])
                        cv2.putText(score_print,'confidence: '+str(score_j)[:6],(candidates[i_][0],candidates[i_][1]),cv2.FONT_HERSHEY_PLAIN, 1.0, 122, 2)
                        cv2.putText(score_print,'nums: '+str(number_thre)[:6],(candidates[j_][0],candidates[j_][1]),cv2.FONT_HERSHEY_PLAIN, 1.0, 80, 2)
                        # cv2.imwrite('./1.jpg', score_print)

    return temo_predict, points, score_print


def get_test_data_rgb(test_ims_, Path, onnx_session, test_data_seg_rgb):

    test_rgb = dict()
    for ind, i in enumerate(test_ims_):
        if i[-3:] in ['png', 'jpg', 'bmp']:
            img_path = os.path.join(Path, i)

            tmp = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            cv2.imwrite('./ycy.jpg', tmp)
            img = cv2.imread('./ycy.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(1024, 1024),interpolation=cv2.INTER_LINEAR)

            img_ = sdk_pre(img)
            onnx_inputs = {onnx_session.get_inputs()[0].name:img_.astype(np.float32)}
            onnx_predict = onnx_session.run(None, onnx_inputs)
            predict = softmax(onnx_predict[0],1)
            map_, points, mask_map = sdk_post(onnx_predict, predict)
            m = map_.astype(np.uint8)
            color = [np.round(a, 2) for a in img[m != 0].mean(axis=0)]
            test_rgb[test_ims[ind]] = color

    data = json.dumps(test_rgb)
    with open(test_data_seg_rgb, 'w') as js_file:
        js_file.write(data)


if __name__ == "__main__":
    onnx_path = './10000.onnx'
    Path = r"D:\work\project\卡尔蔡司膜色缺陷\22.01.7data\all_data\2022-1-7采图\lower_light1"
    test_ims = open('/data/home/jiachen/project/zeiss_mose_seg_roi/help_code/1209test.txt', 'r').readlines()[0].split(',')[:-1]
    test_ims_ = ["1209_{}.bmp".format(a) for a in test_ims]

    test_data_seg_rgb = './test_data_seg_rgb.json'

    onnx_session = ort.InferenceSession(onnx_path)
    get_test_data_rgb(test_ims_, Path, onnx_session, test_data_seg_rgb)








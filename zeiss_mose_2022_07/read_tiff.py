# coding=utf-8
from harvesters.core import Harvester

# img_path1 = r"D:\mac_air_backup\chenjia\Download\Smartmore\2022\膜色缺陷\bits_test0221\bits_test\12bit.bmp"
# img_path2 = r"D:\mac_air_backup\chenjia\Download\Smartmore\2022\膜色缺陷\bits_test0221\bits_test\8bit.bmp"
#img_path3 = r'C:\Users\15974\Desktop\1\1.tiff'

# img_path3 = r'C:\Users\15974\Desktop\1\1.bmp'   # +0.50 0.00.bmp'

# h = Harvester()
# h.add_file(img_path1)

# print(h.files)

# h.update()
# print(h.device_info_list)

# ia = h.create_image_acquirer(serial_number='U430004')
# print("Image Acquirer created")

# Set PixelFormat to BGR12p
# ia.remote_device.node_map.PixelFormat.value = 'BGR12p'
# print('Pixel Format = {0}'.format(ia.remote_device.node_map.PixelFormat.value))
# print("Launching Second Acquisition")
# ia.start_acquisition()
# with ia.fetch_buffer() as buffer:
#     component = buffer.payload.components[0]
#     print(type(component))
#     _1d = component.data
#     print(_1d)
#     ia.stop_acquisition()
# print("Acquisition stopped")

# ia.destroy()
# h.reset()
# print("Finished")

from skimage import io
from PIL import Image
import numpy as np

# im = io.imread(img_path)
# print(im.shape)
# im1 = io.imread(img_path2)
# print(im1[33][33])

# im1 = Image.open(img_path3)
# img = im1.tobytes()


from matplotlib import image as imgplt
# im = imgplt.imread(img_path3)
# print(im[0][0])

# from osgeo import gdal,ogr,osr
# dataset = gdal.Open(img_path3)


# from libtiff import TIFF
import numpy as np
from scipy import misc
from PIL import Image

# tif32 = misc.imread(img_path3) 
# print(tif32[0][0])




# okk code  解析16位图像, 可获取像素的浮点精度
import cv2
def cv_imread_by_np(filePath, clr_type=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.float32), clr_type)

    return cv_img

img_path3 = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\膜色缺陷\mose_0705\mose\RGB\+0.50 0.00.tiff'
# img_data = cv2.imdecode(np.fromfile(img_path3, dtype=np.float32), -1)
img_data = cv_imread_by_np(img_path3, clr_type=cv2.IMREAD_UNCHANGED)
print('max value: {}'.format(np.max(img_data)))
value = img_data[0][0] 
print(img_data.dtype)
changed_value = [255 * float(a/(2**16)) for a in value]
print(changed_value)


# 默认float64数据类型
test_img = np.zeros(img_data.shape)
h, w, _ = img_data.shape[:3]
for i in range(h):
    for j in range(w):
        test_img[i][j][0] = 255 * float(img_data[i][j][0]/(2**16))
        test_img[i][j][1] = 255 * float(img_data[i][j][1]/(2**16))
        test_img[i][j][2] = 255 * float(img_data[i][j][2]/(2**16))

cv2.imwrite(r'C:\Users\15974\Desktop\1\2.jpg', test_img)


# test rgb value
img_path = r'D:\mac_air_backup\chenjia\Download\Smartmore\2022\膜色缺陷\mose0712\1-RGB\1_1.tiff'
img_data = cv_imread_by_np(img_path, clr_type=cv2.IMREAD_UNCHANGED)
print(img_data[600][668])
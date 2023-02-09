from BaseDT.data import ImageData
from BaseDT.plot import imshow_det_bboxes
import numpy as np
# import mmcv
# #
# mmcv.imshow_bboxes()

if __name__ == "__main__":
    #img = cv2.imread("D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg")
    img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg"
    data = ImageData(img, size=(256, 256), crop_size=(224,224))
    imshow_det_bboxes(img, np.array([[56,56,256,256,1]]),np.array(["car"]))
    # print(data.value.shape)
    # print(data.raw_value.shape)
    # tensor_value = data.to_tensor()
    #print(tensor_value)
    # texts = {'city': 'Dubai', 'temperature': 33},
    # data = TextData(texts, vectorize = True)
    # print(data.value)
# img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg"
# data = ImageData(img, size=(256, 256), crop_size=(224,224))
# input_data = np.array()
#
#
# img_data = input_data.astype('float32')
# mean_vec = np.array([0.485, 0.456, 0.406])
# stddev_vec = np.array([0.229, 0.224, 0.225])
# norm_img_data = np.zeros(img_data.shape).astype('float32')
# for i in range(img_data.shape[0]):
#     norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
# norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
#
# norm_img_data = ImageData(input_data, size=(256, 256), crop_size=(224,224))


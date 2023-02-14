from BaseDT.data import ImageData
from BaseDT.io import *
import matplotlib.pyplot as plt
from BaseDT.plot import imshow_det_bboxes
import numpy as np
# import mmcv
# #
# mmcv.imshow_bboxes()

if __name__ == "__main__":
    #img = cv2.imread("D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg")
    # img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg"
    # data = ImageData(img, size=(256, 256), crop_size=(224,224))
    # data.show()
    #imshow_det_bboxes(img, [[56,56,256,256,1]],[0], ["cat"], 0.5)
    # print(data.value.shape)
    # print(data.raw_value.shape)
    # tensor_value = data.to_tensor()
    #print(tensor_value)
    # texts = {'city': 'Dubai', 'temperature': 33},
    # data = TextData(texts, vectorize = True)
    # print(data.value)
    mic = MicroPhone()
    plt.plot(mic.record_audio())
    plt.show()

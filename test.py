from BaseDT.data import ImageData, TextData
from BaseDT.io import *
import matplotlib.pyplot as plt
from BaseDT.plot import imshow_det_bboxes
import numpy as np
# import mmcv
# #
# mmcv.imshow_bboxes()

if __name__ == "__main__":
    # img = cv2.imread("D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg")
    img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg"
    data = ImageData(img, backbone = "MobileNet")
    # print(data.value.shape)
    # data.show()
    data = ImageData(img, size=(256, 256), crop_size=(224, 224), normalize=False)
    data = ImageData(img, crop_size=(224,224), size=(256, 256), normalize=False)
    data.show()
    #imshow_det_bboxes(img, [[56,56,256,256,1],[16,16,156,156,1]],[0,1], ["cat","dog"], 0.5)
    # print(data.value.shape)
    # print(data.raw_value.shape)
    # tensor_value = data.to_tensor()
    #print(tensor_value)
    # texts = {'city': 'Dubai', 'temperature': 33}
    # # texts = ['<><\b>', 'This is the second second document.', 'And the third one.', 'Is this the first document?', ]
    # data = TextData(texts, vectorize = True)
    # print(data.value)
    # mic = MicroPhone()
    # plt.plot(mic.record_audio())
    # plt.show()




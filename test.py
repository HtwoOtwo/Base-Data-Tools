from BaseDT.data import ImageData, TextData
from BaseDT.dataset import DataSet
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
    ds = DataSet(r"C:\Users\76572\Desktop\my_dataset")
    ds.make_dataset(r"C:\Users\76572\Desktop\Rabbits_coco", src_format="coco")
    # ds.move_files(r"C:\Users\76572\Desktop\Rabbits_voc\train", r"C:\Users\76572\Desktop\Rabbits_voc\annotations", '.xml')
    # ds.check()
    # ds.convert_data_to_coco_format(r"C:\Users\76572\Desktop\AILab\xedu\dataset\det\cats_and_dogs")
    # ds.rename_files_in_coco(r"D:\PythonProject\OpenMMLab-Edu-main\dataset\det\Rabbits\annotations\valid.json", r"D:\PythonProject\OpenMMLab-Edu-main\dataset\det\Rabbits\images\val_set")





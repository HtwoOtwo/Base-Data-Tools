# 导入工具库
import numpy as np
import matplotlib.pyplot as plt
import cv2
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numbers

class ImageData(object):
    '''
    目前支持图片路径, np array, 文本
    后续支持 PIL
    '''
    _defaults = {
        "data_type": 'uint8',
        # 图像默认属性
        "to_rgb": True,
        "normalize": True,
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
    }
    _backbone_args = {
        "LeNet": {"size": (32, 32), "to_rgb": False, "mean": [33.46], "std":[78.87]},
        "ResNet18": {"size": (256, 256), "crop_size": (224, 224)},
        "ResNet50": {"size": (256, 256), "crop_size": (224, 224)},
        "ResNeXt": {"size": (256, 256), "crop_size": (224, 224)},
        "ShuffleNet_v2": {"size": (256, 256), "crop_size": (224, 224)},
        "VGG": {"size": (256, 256), "crop_size": (224, 224)},
        "RepVGG": {"size": (256, 256), "crop_size": (224, 224)},
        "MobileNet": {"size": (256, 256), "crop_size": (224, 224)},
        "SSD_Lite": {"size": (320, 320), "pad_size_divisor": 320},
        "FasterRCNN": {"size": (1333, 800), "size_keep_ratio": True,  "pad_size_divisor": 32},
        "Mask_RCNN": {"size": (1333, 800), "size_keep_ratio": True, "pad_size_divisor": 32},
        "RegNet": {"size": (1333, 800), "size_keep_ratio": True, "pad_size_divisor": 32},
        "Yolov3": {"size": (416, 416), "size_keep_ratio": True, "pad_size_divisor": 32},
    }

    def get_attribute(self, n):
        if n in self.__dict__:
            return self.__dict__[n]
        else:
            return None

    def __init__(self, data_source, **kwargs):
        self.vct = None
        self.data_source = data_source
        # 装在默认参数进入私有变量列表
        self.__dict__.update(self._defaults)
        # 将关键词参数装入私有变量列表
        for name, value in kwargs.items():
            setattr(self, name, value)

        if type(self.data_source) == str:
            self.value = cv2.imdecode(np.fromfile((self.data_source),dtype=np.uint8),-1)
        if type(self.data_source) == np.ndarray:
            self.value = self.data_source
        else:
            # TODO 检查合法性
            pass
        self.value = self.value.astype(self.get_attribute("data_type"))
        self.raw_value = self.value
        if self.get_attribute("backbone"):
            # for key, value in self._backbone_args[self.get_attribute("backbone")].items():
            #     print(key,value)
            self.value = ImageData(data_source, **self._backbone_args[self.get_attribute("backbone")]).value
            self.__dict__.update(self._backbone_args[self.get_attribute("backbone")])#更新私有变量列表
        else:
            if self.get_attribute("to_rgb") == False and len(self.value.shape)>=3:
                self._rgb2gray()
            elif self.get_attribute("to_rgb") == True:
                self._to3channel()
            #需要考虑各种操作顺序的灵活性，循环遍历私有变量列表来实现
            for key in self.__dict__.keys():
                if key == "size" or key == "size_keep_ratio":
                    size = self.get_attribute("size")
                    size_keep_ratio = self.get_attribute("size_keep_ratio")
                    self._resize(size, size_keep_ratio)
                if key == "crop_size":
                    crop_size = self.get_attribute("crop_size")
                    self._crop(crop_size)
                if key == "normalize":
                    #TODO 需检查维度
                    if self.get_attribute("normalize") == True:
                        mean = self.get_attribute("mean")
                        std = self.get_attribute("std")
                        self._normalize(mean, std)
                if key == "pad_size_divisor":
                    pad_size_divisor = self.get_attribute("pad_size_divisor")
                    self._pad(size_divisor=pad_size_divisor)

    def to_tensor(self):
        if self.get_attribute("to_rgb"):
            return np.expand_dims(np.transpose(self.value, (2,0,1)), 0)
        else:
            return np.expand_dims(np.expand_dims(self.value, 0), 0)


    #保护方法，不给用户调用
    def _rgb2gray(self):
        gray = np.dot(self.value, [0.2989, 0.5870, 0.1140])
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        self.value = gray

    def _to3channel(self, background=(255, 255, 255)):
        if len(self.value.shape) == 2:
            self.value = cv2.cvtColor(self.value, cv2.COLOR_GRAY2RGB)
        elif len(self.value.shape) == 3 and self.value.shape[-1] > 3:
            row, col, ch = self.value.shape
            rgb = np.zeros((row, col, 3), dtype='float32')
            r, g, b, a = self.value[:, :, 0], self.value[:, :, 1], self.value[:, :, 2], self.value[:, :, 3]
            a = np.asarray(a, dtype='float32') / 255.0
            R, G, B = background
            rgb[:, :, 0] = r * a + (1.0 - a) * R

            rgb[:, :, 1] = g * a + (1.0 - a) * G

            rgb[:, :, 2] = b * a + (1.0 - a) * B
            self.value = np.asarray(rgb, dtype='uint8')

    def _normalize(self, mean, std):
        # assert self.value.dtype != np.uint8
        img = self.value
        mean = np.float64(np.array(mean).reshape(1, -1))
        stdinv = 1 / np.float64(np.array(std).reshape(1, -1))
        if self.get_attribute("to_rgb"):
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        img = img.astype(np.float32)
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        self.value = img

    def _resize(self, size, keep_ratio = False):
        #TODO 支持padding
        if keep_ratio:
            h, w = self.value.shape[:2]
            img_shape = self.value.shape[:2]
            max_long_edge = max(size)
            max_short_edge = min(size)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
            # scale_factor = [1.0*size[0]/w, 1.0*size[1]/h]
            new_size = self._scale_size(img_shape[::-1], scale_factor)
            # print(new_size)
            self.value = cv2.resize(self.value, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            self.value = cv2.resize(self.value, size, interpolation=cv2.INTER_LINEAR)
        #self.value = np.resize(self.value, size, interp='bilinear')

    def _scale_size(self, size, scale):
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        w, h = size
        return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

    def _bbox_clip(self, bboxes, img_shape):
        assert bboxes.shape[-1] % 4 == 0
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
        return clipped_bboxes

    def _bbox_scaling(self, bboxes, scale, clip_shape=None):
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.copy()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
        if clip_shape is not None:
            return self._bbox_clip(scaled_bboxes, clip_shape)
        else:
            return scaled_bboxes

    def _crop(self, size, scale=1.0, pad_fill=None):
        if isinstance(size, int):
            crop_size = (size, size)
        else:
            crop_size = size

        img = self.value
        img_height, img_width = img.shape[:2]

        crop_height, crop_width = crop_size

        if crop_height > img_height or crop_width > img_width:
            #TODO 可选择pad_mod
            pass
        else:
            crop_height = min(crop_height, img_height)
            crop_width = min(crop_width, img_width)

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        chn = 1 if img.ndim == 2 else img.shape[2]
        if pad_fill is not None:
            if isinstance(pad_fill, (int, float)):
                pad_fill = [pad_fill for _ in range(chn)]
            assert len(pad_fill) == chn

        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = self._bbox_scaling(_bboxes, scale).astype(np.int32)
        clipped_bbox = self._bbox_clip(scaled_bboxes, img.shape)

        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
            if pad_fill is None:
                patch = img[y1:y2 + 1, x1:x2 + 1, ...]
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
                if chn == 1:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
                else:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
                patch = np.array(
                    pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                patch[y_start:y_start + h, x_start:x_start + w,
                ...] = img[y1:y1 + h, x1:x1 + w, ...]
            patches.append(patch)

        if bboxes.ndim == 1:
            self.value = patches[0]
        else:
            self.value = patches

    def _pad(self, size=None, size_divisor=None, pad_val=0, padding_mode='constant'):
        img = self.value
        if size_divisor is not None:
            if size is None:
                size = (img.shape[0], img.shape[1])
            pad_h = int(np.ceil(
                size[0] / size_divisor)) * size_divisor
            pad_w = int(np.ceil(
                size[1] / size_divisor)) * size_divisor
            size = (pad_h, pad_w)

        shape = size
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                             f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)
        self.value = img

    def _flip(self):
        pass

    def show(self, raw = False):
        if raw: img = self.raw_value
        else: img = self.value
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    def map_orig_coords(self, boxes):
        #TODO 暂时只支持box的映射，后续应该支持单纯的坐标映射
        original_w, original_h = self.raw_value.shape[:2]
        processed_w, processed_h = self.value.shape[:2]
        sx, sy = original_w / processed_w, original_h / processed_h
        new_boxes = np.array([]).reshape(0, 5)
        for box in boxes:
            new_box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy, box[4]]
            new_boxes = np.concatenate([new_boxes, [new_box]], axis=0)
        return new_boxes

class TextData(object):
    _defaults = {
        "data_type": 'uint8',
        # 文本默认属性
        "vct": {"str": CountVectorizer(), "dict": DictVectorizer()},
    }

    def get_attribute(self, n):
        if n in self.__dict__:
            return self.__dict__[n]
        else:
            return None

    def __init__(self, data_source, **kwargs):
        self.vct = None
        self.data_source = data_source
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 这里的data_source不可以是文件
        self.raw_value = self.data_source
        self.value = self.raw_value
        if type(self.raw_value) == dict or type(self.raw_value[0]) == dict:
            self.text_type = "dict"
            if type(self.raw_value) == dict: self.value = [self.raw_value]
        elif type(self.raw_value) == str or type(self.raw_value[0]) == str:
            self.text_type = "str"
            if type(self.raw_value) == str: self.value = [self.raw_value]
        else:
            # TODO 检查合法性
            pass
        if self.get_attribute("vectorize") == True:
            self.value = self.vectorizer(self.get_attribute("chinese"), self.get_attribute("fitted"))

    def vectorizer(self, chinese = False, fitted = False):
        '''
        支持单文本及多文本
        :param text: 允许raw txt输入以及半处理数据例如字典输入
        :return: 词向量矩阵（词向量组成的矩阵）
        '''
        txt_cnt = None
        text_type = self.text_type
        texts = self.value

        if chinese:
            for i in range(len(texts)):
                texts[i] = self._chinese_cut(texts[i])
        #过滤
        # for i in range(len(texts)):
        #     texts[i] = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9\sP]+', '', texts[i])
        if text_type == "dict":
            if not fitted: txt_cnt = self.vct[text_type].fit_transform(texts)
            if fitted: txt_cnt = self.vct[self.text_type].transform(texts)

        if text_type=="str":
            if not fitted: txt_cnt = self.vct[text_type].fit_transform(texts)
            if fitted: txt_cnt = self.vct[text_type].transform(texts)
        return txt_cnt.A

    def fit(self, texts = ""):
        '''
        生成词汇表，支持单文本及多文本
        :param text:
        :return:
        '''
        text_type = self.text_type
        self.vct[text_type] = self.vct[text_type].fit(texts)

    def get_feature_names(self):
        texts = self.value
        text_type = self.text_type
        return self.vct[text_type].get_feature_names_out()

    def _chinese_cut(self, text):
        '''
        单文本操作
        :param text:
        :return: 返回字符串
        '''
        words = list(jieba.cut(text))
        text = "/ ".join(words)
        # text = "".join(words)
        return text

    def to_tensor(self):
        pass
   
class ModelData(object):

    def __init__(self, path):
        self.path = path

    def get_labels(self):
        assert isinstance(self.path, str)
        file_path = self.path
        class_name = []
        key = 'CLASSES'
        if file_path[-3:] == 'txt':
            with open(file_path, 'r') as f:
                class_name = [tag.strip() for tag in f.readlines()]
        elif file_path[-3:] == 'pth':
            import torch
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))
            if key in state_dict['meta']:
                class_name = state_dict['meta'][key]
        elif file_path[-4:] == 'onnx':
            import onnxruntime
            sess = onnxruntime.InferenceSession(file_path)
            model_meta = sess.get_modelmeta()
            if key in model_meta.custom_metadata_map:
                unicode_string = model_meta.custom_metadata_map[key]
                class_name = unicode_string.split('|')
        else:
            file_allow = ['txt', 'pth', 'onnx']
            print('This function currently supports file formats in {}, please check the path'.format(file_allow))
        if len(class_name) == 0:
            print('Class_name in model is empty. Please use model generate by MMEdu')
        return class_name

import numpy as np
import cv2
import os
import pickle
import json
import urllib.request
from subprocess import Popen, TimeoutExpired, PIPE
import subprocess

class DataSet(object):
    _defaults = {
        "Middlebury_2001": ["echo Y | odl get Middlebury_2001\n"],
        "Letter": ["echo Y | odl get Letter\n"],
    }
    def __init__(self, dataset_path = None):
        if dataset_path is None: self.dataset_path = os.getcwd()
        else: self.dataset_path = dataset_path

    def download(self, dataset_name):
        username = "luyanan@pjlab.org.cn"
        #TODO 密码保护，加密
        password = "*********"
        original_directory = os.getcwd()
        os.chdir(self.dataset_path)
        os.system("odl login -u {} -p {}".format(username, password))
        if dataset_name not in self._defaults.keys():
            print("This dataset is not download supported, you can try:")
            print(list(self._defaults.keys()))
        else:
            for command in self._defaults[dataset_name]:
                os.system(command)
        os.chdir(original_directory)

    def readTag(self, path=None):
        if type(path) == str and path.endswith("txt"):
            with open(path) as f:
                tag = [e.rstrip('\n') for e in f.readlines()]
        else:
            try:
                with open(os.path.join(self.dataset_path, "classes.txt")) as f:
                    tag = [e.rstrip("\n") for e in f.readlines()]
            except:
                raise ValueError("Dataset directory is missing required classes.txt")
        print(tag)
        return tag


    def check(self):
        pass

    def print_folder_structure(self, root_folder, indent=''):
        # 遍历文件夹内的所有文件和文件夹
        file_count = 0
        files = []
        for item in os.listdir(root_folder):
            # 构造完整的文件路径
            item_path = os.path.join(root_folder, item)
            # 如果是文件夹，递归调用该函数
            if os.path.isdir(item_path):
                print(indent + item + '/')
                self.print_folder_structure(item_path, indent + '  ')
            # 否则，增加文件计数器
            else:
                file_count += 1
                files.append(item)
        # 如果文件数量大于 10，打印文件数量
        if file_count > 10:
            print(indent + f'({file_count} files)')
        else:
            for file in files:
                print(indent + file)


    def check_imagenet_format(self, dataset_path = None):
        if dataset_path == None: dataset_path = self.dataset_path
        # 检查文件夹是否存在
        if not os.path.exists(dataset_path):
            raise ValueError("Dataset directory does not exist")

        if not os.path.exists(os.path.join(dataset_path,"classes.txt")):
            raise ValueError("Dataset directory is missing required classes.txt")

        # 检查子文件夹是否存在
        required_subdirs = ["training_set", "val_set", "test_set"]
        with open(os.path.join(dataset_path,"classes.txt")) as f:
            required_subsubdirs = [e.rstrip("\n") for e in f.readlines()]
        for subdir in required_subdirs:
            for subsubdir in required_subsubdirs:
                subdir_path = os.path.join(dataset_path, subdir, subsubdir)
                if not os.path.exists(subdir_path):
                    raise ValueError("Dataset directory is missing required subdirectory: {}".format(subdir))

        # 检查数据量是否足够
        required_num_files = {
            "training_set": 100,
            "val_set": 10,
            # "test_set": 10
        }
        for subdir, num_files in required_num_files.items():
            for subsubdir in required_subsubdirs:
                subdir_path = os.path.join(dataset_path, subdir, subsubdir)
                num_files_in_subdir = len([f for f in os.listdir(subdir_path) if f.endswith(".jpg") or f.endswith(".png")])
                if num_files_in_subdir < num_files:
                    raise ValueError("Subdirectory {} has fewer than {} image files".format(subdir, num_files))
        print("Dataset is in ImageNet format")

    def check_coco_format(self, dataset_path = None):
        if dataset_path == None: dataset_path = self.dataset_path
        if not os.path.exists(os.path.join(dataset_path, 'annotations')):
            raise ValueError('Annotations folder does not exist')

        # 检查train和val文件夹是否存在
        # TODO 这里的test应该是val
        if not (os.path.exists(os.path.join(dataset_path, 'images/train')) or os.path.exists(
                os.path.join(dataset_path, 'images/test'))):
            raise ValueError('train or test folder does not exist')

        # Get the paths to the annotation files
        annotation_paths = [os.path.join(dataset_path, 'annotations', file) for file in
                            os.listdir(os.path.join(dataset_path, 'annotations'))]

        # 检查标注文件
        for annotation_path in annotation_paths:
            # Load the annotation file
            with open(annotation_path, 'r') as file:
                annotation = json.load(file)

            # 检查标注文件格式是否正确
            if 'annotations' in annotation and 'categories' in annotation and 'images' in annotation:
                #print('Annotation file is in the correct format')
                pass
            else:
                raise ValueError('Annotation file is not in the correct format')

            images = annotation['images']

            # 检查图片
            for image in images:
                # 检查图片是否有标注
                if 'id' in image and any(ann['image_id'] == image['id'] for ann in annotation['annotations']):
                    #print('Image has a corresponding annotation')
                    pass
                else:
                    raise ValueError('Image does not have a corresponding annotation')
        print("Dataset is in COCO format")


    def rename_files_in_coco(self, annotations_file, images_dir):
        '''
        有些文件名太长，或者文件损坏，此函数可以检查文件并重新命名为0001，0002.....，损坏文件将被自动删除
        :param annotations_file:
        :param images_dir:
        :return:
        '''
        # 读取 annotation 文件
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        # 遍历 annotation 文件中的图像文件名
        n_images = len(annotations)
        id_to_remove = []
        for i, annotation in enumerate(annotations["images"]):
            old_filename = annotation['file_name']
            file_ext = os.path.splitext(old_filename)[1]
            new_filename = f"{i+1:05d}".format(i) + file_ext
            #new_filename = f"{i + 1:05d}".format(i) + file_ext
            old_path = os.path.join(images_dir, old_filename)
            new_path = os.path.join(images_dir, new_filename)
            # Use long file name
            old_path = os.path.abspath(os.path.join('\\\\?\\', os.path.abspath(old_path)))
            new_path = os.path.abspath(os.path.join('\\\\?\\', os.path.abspath(new_path)))
            # Rename the file
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                print(f"Error : can't rename {old_path}, due to :{e}")
                id_to_remove.append(annotation['id'])
            # Update the annotation file
            annotations["images"][i]['file_name'] = new_filename
        # 保存 annotation 文件
        new_annotations_images = [anno for anno in annotations["images"] if anno['id'] not in id_to_remove]
        new_annotations_annotations = [anno for anno in annotations["annotations"] if
                                       anno['image_id'] not in id_to_remove]
        annotations["images"] = new_annotations_images
        annotations["annotations"] = new_annotations_annotations
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)


if __name__ == "__main__":
    # Example usage
    # dataset_dir = r"C:\Users\76572\Desktop\AILab\xedu\dataset\det\coco"
    # # dataset_dir = r"C:\Users\76572\Desktop\AILab\xedu\dataset\cls\hand_gray"
    # ds = DataSet(dataset_dir)
    # #ds.readTag(r"C:\Users\76572\Desktop\AILab\xedu\dataset\cls\CatsDogs\classes.txt")
    # #ds.check_imagenet_format()
    # ds.check_coco_format()
    # # ds.print_folder_structure(dataset_dir)

    ds = DataSet(r"D:\PythonProject\OpenMMLab-Edu-main\dataset\det\Rabbits")
    ds.rename_files_in_coco(r"D:\PythonProject\OpenMMLab-Edu-main\dataset\det\Rabbits\annotations\valid.json", r"D:\PythonProject\OpenMMLab-Edu-main\dataset\det\Rabbits\images\val_set")





import numpy as np
import cv2
import os
import pickle
import json
import urllib.request
from subprocess import Popen, TimeoutExpired, PIPE
import subprocess
import shutil
import random
import xml.etree.ElementTree as ET

class DataSet(object):
    _defaults = {
        "Middlebury_2001": ["echo Y | odl get Middlebury_2001\n"],
        "Letter": ["echo Y | odl get Letter\n"],
    }
    def __init__(self, dataset_path = None, dataset_type = None):
        self.dataset_type = dataset_type
        if dataset_path is None: self.dataset_path = os.getcwd()
        else: self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            #print("数据集文件夹不存在，已为您创建")
            os.makedirs(dataset_path)

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

    def read_json(self, file_path):
        with open(file_path, 'rb') as jsonfile:
            return json.load(jsonfile)

    def write_json(self, data, file_path):
        with open(file_path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=1)

    def copy_images(self, src_path, dst_path):
        has_subdir = False
        src_subfolders = os.listdir(src_path)
        for item in src_subfolders:
            item_path = os.path.join(src_path, item)
            if os.path.isdir(item_path):
                has_subdir =  True
                break
        # 若无子文件夹则，在外套一个子文件夹
        if not has_subdir:
            src_subfolders = [os.path.basename(src_path)]
            src_path = os.path.dirname(src_path)
        for subfolder_name in src_subfolders:
            src_subfolder_path = os.path.join(src_path, subfolder_name)
            if os.path.isdir(src_subfolder_path):
                dst_subfolder_path = os.path.join(dst_path, subfolder_name)
                if not os.path.exists(dst_subfolder_path):
                    os.makedirs(dst_subfolder_path)
                for img in os.listdir(src_subfolder_path):
                    # 复制jpg和png图片
                    if img.endswith('.jpg') or img.endswith('.png'):
                        src_img_path = os.path.join(src_subfolder_path, img)
                        dst_img_path = os.path.join(dst_subfolder_path, img)
                        # 复制图片
                        shutil.copy(src_img_path, dst_img_path)

        print("图片已复制成功")


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


    def check_cls(self, dataset_path = None):
        if dataset_path == None: dataset_path = self.dataset_path
        # 检查文件夹是否存在
        if not os.path.exists(dataset_path):
            raise ValueError("数据集路径不存在")

        if not os.path.exists(os.path.join(dataset_path,"classes.txt")):
            raise ValueError("classes.txt缺失")
        if not os.path.exists(os.path.join(dataset_path,"val.txt")):
            raise ValueError("val.txt缺失")

        # 检查子文件夹是否存在
        required_subdirs = ["training_set", "val_set"]
        with open(os.path.join(dataset_path,"classes.txt")) as f:
            required_subsubdirs = [e.rstrip("\n") for e in f.readlines()]
        for subdir in required_subdirs:
            for subsubdir in required_subsubdirs:
                subdir_path = os.path.join(dataset_path, subdir, subsubdir)
                if not os.path.exists(subdir_path):
                    raise ValueError("{}文件夹缺失".format(subsubdir))

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
                    print("{} 文件夹图片数量不足 {} 可能会影响训练结果".format(subdir, num_files))
        print("数据集符合cls标准")

    def check_det(self, dataset_path = None):
        if dataset_path == None: dataset_path = self.dataset_path
        if not os.path.exists(os.path.join(dataset_path, 'annotations')):
            raise ValueError('Annotations文件夹不存在')
        if not os.path.exists(os.path.join(dataset_path, 'classes.txt')):
            raise ValueError('classes.txt文件夹不存在')

        # 检查train和val文件夹是否存在
        required_subdirs = ["train", "valid"]
        for subdir in required_subdirs:
            subdir_path = os.path.join(dataset_path, "images", subdir)
            if not os.path.exists(subdir_path):
                raise ValueError("{}文件夹缺失".format(subdir))
            if not os.path.exists(os.path.join(dataset_path, "annotations", "{}.json".format(subdir))):
                raise ValueError("{}.json文件缺失".format(subdir))

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
                raise ValueError('json文件格式错误')

            images = annotation['images']

            # # 检查图片
            # for image in images:
            #     # 检查图片是否有标注
            #     if 'id' in image and any(ann['image_id'] == image['id'] for ann in annotation['annotations']):
            #         #print('Image has a corresponding annotation')
            #         pass
            #     else:
            #         raise ValueError('图片未正确标注')

        # 检查数据量是否足够
        required_num_files = {
            "train": 100,
            "valid": 10,
            # "test_set": 10
        }
        required_subdirs = ["train", "valid", "test"]
        for subdir, num_files in required_num_files.items():
            subdir_path = os.path.join(dataset_path, "images", subdir)
            num_files_in_subdir = len(
                [f for f in os.listdir(subdir_path) if f.endswith(".jpg") or f.endswith(".png")])
            if num_files_in_subdir < num_files:
                print("{} 文件夹图片数量不足 {} 可能会影响训练结果".format(subdir, num_files))
        print("数据集符合det标准")


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

    def find_json(self, dir_path):
        json_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files

    def find_xml(self, dir_path):
        xml_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))
        return xml_files


    def make_dataset(self, source, src_format = "innolab", train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2):
        input_dir = source
        output_dir = self.dataset_path
        try:
            if src_format.upper() == "INNOLAB":
                self.innolab2coco(input_dir, output_dir, train_ratio, test_ratio, val_ratio)
                self.check_det()
            elif src_format.upper() == "COCO":
                self.coco2coco(input_dir, output_dir, train_ratio, test_ratio, val_ratio)
                self.check_det()
            elif src_format.upper() == "VOC":
                self.voc2coco(input_dir, output_dir, train_ratio, test_ratio, val_ratio)
                self.check_det()
            elif src_format.upper() == "IMAGENET":
                self.split_dataset(input_dir, output_dir, train_ratio, test_ratio, val_ratio)
                self.check_cls()
            else:
                raise ValueError("未支持的数据集格式")
            self.print_folder_structure(self.dataset_path)
        except Exception as e:
            raise e



    def split_dataset(self, input_dir, output_dir, train_ratio, test_ratio, val_ratio):
        try:
            self._check(input_dir)
        except ValueError as e:
            # ignore ValueError with specific message
            if str(e) == "no annotations":
                pass
            else:
                print(e)
                print("文件夹结构不正确或子文件夹命名错误，正确的文件夹结构为：")
                print("|---images\n\t|----class1\n\t\t|----xxx.jpg/xxx.png/....\n\t|----classN\n\t\t|----xxx.jpg/xxx.png/....\n|---classes.txt")
                return
        # 清空原文件夹
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        # 构建数据集文件夹结构
        train_dir = os.path.join(output_dir, "training_set")
        val_dir = os.path.join(output_dir, "val_set")
        test_dir = os.path.join(output_dir, "test_set")
        for dir in [train_dir, val_dir, test_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
            for subdir in os.listdir(os.path.join(input_dir, "images")):
                subdir_path = os.path.join(dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)

        # 读取类别信息文件
        classes_path = os.path.join(input_dir, "classes.txt")
        with open(classes_path, "r") as f:
            classes = f.read().splitlines()

        # 将每个类别与一个唯一的整数标签对应起来
        class_to_label = {classes[i]: i for i in range(len(classes))}

        # 遍历每个子文件夹
        print("正在划分数据集，比例为 train:test:val = {}:{}:{}".format(train_ratio, test_ratio, val_ratio))
        print("转换中......")
        for subdir in os.listdir(os.path.join(input_dir, "images")):
            subdir_path = os.path.join(input_dir, "images", subdir)
            file_list = os.listdir(subdir_path)
            num_files = len(file_list)
            random.shuffle(file_list)  # 打乱文件列表顺序

            # 计算划分后每个集合的文件数目
            num_train_files = int(num_files * train_ratio)
            num_val_files = int(num_files * val_ratio)
            num_test_files = num_files - num_train_files - num_val_files

            # 划分数据集并拷贝到对应文件夹中
            for i, file in enumerate(file_list):
                file_path = os.path.join(subdir_path, file)
                if i < num_train_files:
                    dst_dir = os.path.join(train_dir, subdir)
                elif i < num_train_files + num_val_files:
                    dst_dir = os.path.join(val_dir, subdir)
                else:
                    dst_dir = os.path.join(test_dir, subdir)
                shutil.copy(file_path, dst_dir)

        # 生成train.txt, val.txt, test.txt文件
        for split in [("val","val_set"), ("test","test_set")]:
            with open(os.path.join(output_dir, split[0] + ".txt"), "w") as f:
                for subdir in os.listdir(os.path.join(output_dir, split[1])):
                    subdir_path = os.path.join(output_dir, split[1], subdir)
                    for file in os.listdir(subdir_path):
                        file_path = subdir + '/' + file
                        class_name = subdir
                        label = class_to_label[class_name]
                        f.write(file_path + " " + str(label) + "\n")
        shutil.copy(classes_path, output_dir)
        print("转换成功")


    def innolab2coco(self, input_dir, output_dir, train_ratio, test_ratio, val_ratio):
        '''
        由于读写权限，classes信息直接传入coco2coco
        '''
        classes = []

        labels_json = os.path.join(input_dir, 'labels.json')
        used_labels_json = os.path.join(input_dir, 'used_labels.json')

        ann_out = {"images": [], "type": "instances", "annotations": [], "categories": []}
        cid = {}

        files = self.read_json(labels_json)
        for id, c in enumerate(files):
            ann_out["categories"].append({"supercategory": "None", "id": id, "name": c["name"]})
            cid[c["id"]] = id
            classes.append(c["name"])

        annotation_train_id = 0
        img_train_id = 0

        files = self.read_json(used_labels_json)
        for id, f in enumerate(files):
            p = input_dir + f['filePath'].split('.')[0] + '.json'
            if not os.path.exists(p):
                #print("路径：" + p + " json标注文件不存在，已跳过该对应路径标注文件")
                continue
            single = self.read_json(p)

            s1 = json.loads(single['rectTool'])['step_1']['result']
            s2 = json.loads(single['rectTool'])
            num = len(s1)
            s1 = s1[0]
            name = f['filePath'].split('/')[-1]
            ann_out["images"].append(
                {"file_name": str(name), "height": s2["height"], "width": s2["width"], "id": img_train_id})
            for i in range(num):
                s = json.loads(single['rectTool'])['step_1']['result'][i]
                ann_out["annotations"].append(
                    {"id": annotation_train_id, "image_id": img_train_id, "ignore": 0,
                     "category_id": cid[s["attribute"]],
                     "area": int(s["height"]) * int(s["width"]), "iscrowd": 0,
                     "bbox": [int(s["x"]), int(s["y"]),
                              int(s["width"]), int(s["height"])]})
                annotation_train_id += 1
            img_train_id += 1

        self.coco2coco(input_dir, output_dir, train_ratio, test_ratio, val_ratio, ann_json=ann_out, classes = classes)

    def coco2coco(self, input_dir, output_dir, train_ratio, test_ratio, val_ratio, ann_json = None, classes = None):
        '''
        coco转XEdu coco
        '''
        if ann_json == None:
            try:
                self._check(input_dir)
            except Exception as e:
                if classes == None:
                    print(e)
                    print("文件夹结构不正确或子文件夹命名错误，正确的文件夹结构为：")
                    print("|---annotations\n\t|----xxx.json/xxx.xml/xxx.txt\n|---images\n\t|----xxx.jpg/xxx.png/....\n|---classes.txt")
                raise e
        # 清理文件夹
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        annotations_path = os.path.join(input_dir, "annotations")
        images_path = os.path.join(input_dir, "images")
        if os.path.exists(os.path.join(input_dir, "classes.txt")):
            classes_path = os.path.join(input_dir, "classes.txt")
            shutil.copy(classes_path, output_dir)
        elif classes != None:
            with open(os.path.join(output_dir, "classes.txt"),'w') as f:
                for class_name in classes:
                    f.write(class_name+'\n')
        else:
            print("请提供类别信息")

        # 合并json
        if not ann_json:
            # 检查数据集格式
            json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
            if len(json_files) == 0:
                raise ValueError("没有找到json文件，请确认数据集类型是否正确")
            ann_json = self._merge_json(annotations_path)

        # 划分数据集images和json
        print("正在划分数据集，比例为 train:test:val = {}:{}:{}".format(train_ratio, test_ratio, val_ratio))
        train_files, test_files, val_files = self._split_dataset(images_path, train_ratio, test_ratio, val_ratio)
        train_ann, test_ann, val_ann = self._split_json(ann_json, train_files, test_files, val_files)
        files = [train_files, test_files, val_files]
        print("转换中......")
        #写入数据
        for i, sub_path in enumerate(["train", "test", "valid"]):
            folder_path = os.path.join(output_dir, "images" , sub_path)
            os.makedirs(folder_path, exist_ok=True)
            for file_name in files[i]:
                file_path = os.path.join(images_path,file_name)
                dst_path = os.path.join(folder_path, file_name)
                shutil.copy(file_path, dst_path)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        with open(os.path.join(os.path.join(output_dir, "annotations"), "train.json"), "w") as f:
            json.dump(train_ann, f)
        with open(os.path.join(os.path.join(output_dir, "annotations"), "test.json"), "w") as f:
            json.dump(test_ann, f)
        with open(os.path.join(os.path.join(output_dir, "annotations"), "valid.json"), "w") as f:
            json.dump(val_ann, f)
        print("转换成功")



    def voc2coco(self, input_dir, output_dir, train_ratio, test_ratio, val_ratio):
        '''
        由于读写权限，annotation将直接传入coco2coco
        '''
        try:
            self._check(input_dir)
        except Exception as e:
            print(e)
            print("文件夹结构不正确或子文件夹命名错误，正确的文件夹结构为：")
            print("|---annotations\n\t|----xxx.json/xxx.xml/xxx.txt\n|---images\n\t|----xxx.jpg/xxx.png/....\n|---classes.txt")
            raise e

        annotations_path = os.path.join(input_dir, "annotations")
        xml_files = [f for f in os.listdir(annotations_path) if f.endswith('xml')]
        if len(xml_files) == 0:
            raise ValueError("没有找到xml文件，请确认数据集类型是否正确")

        categories = []
        category_id = 0
        with open(os.path.join(input_dir, "classes.txt"), "r") as f:
            for line in f.readlines():
                category_name = line.strip()
                if category_name:
                    categories.append({
                        "id": category_id,
                        "name": category_name,
                        "supercategory": category_name
                    })
                    category_id += 1
        coco_annotation = {
            "categories": categories,
            "images": [],
            "annotations": []
        }

        # Read VOC annotations and create COCO annotations
        annotation_id = 0
        image_id = 0
        for image_filename in os.listdir(os.path.join(input_dir, "annotations")):
            # Read VOC annotation
            tree = ET.parse(os.path.join(input_dir, "annotations", image_filename))
            root = tree.getroot()
            image_filename = image_filename.replace(".xml", ".jpg")
            image_width = int(root.find("size/width").text)
            image_height = int(root.find("size/height").text)

            # Create COCO annotation for each object in the image
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                category_name = obj.find("name").text
                category_id = None
                for category in categories:
                    if category_name == category["name"]:
                        category_id = category["id"]
                        break

                if category_id is None:
                    continue

                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Create COCO annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                }
                coco_annotation["annotations"].append(annotation)
                annotation_id += 1

            # Create COCO image
            coco_annotation["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height
            })
            image_id += 1

        # Save COCO annotation file
        self.coco2coco(input_dir, output_dir, train_ratio, test_ratio, val_ratio, ann_json=coco_annotation)


    def move_files(self, input_dir, output_dir, suffix):
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                # 如果文件后缀为指定的后缀
                if file.endswith(suffix):
                    # 拼接文件路径
                    file_path = os.path.join(root, file)
                    # 剪切文件到目标文件夹中
                    shutil.move(file_path, output_dir)


    def _split_dataset(self, images_path, train_ratio, test_ratio, val_ratio):
        """
        根据比例划分数据集
        """
        # 获取数据集中的所有文件名
        file_names = [f for f in os.listdir(images_path) if os.path.splitext(f)[1].upper() in [".JPG",".PNG",".JPEG",".BMP","TIFF"]]
        # 洗牌以随机分配
        random.shuffle(file_names)

        # 计算数据集大小和划分点
        dataset_size = len(file_names)
        train_split = int(dataset_size * train_ratio)
        test_split = int(dataset_size * (train_ratio + test_ratio))

        # 划分数据集
        train_files = file_names[:train_split]
        test_files = file_names[train_split:test_split]
        val_files = file_names[test_split:]

        return train_files, test_files, val_files

    def _merge_json(self, annotations_path):
        '''
        将annotations_path下的所有json文件合并
        '''
        input_paths = [os.path.join(annotations_path, f) for f in os.listdir(annotations_path) if f.endswith(".json")]

        images = []
        annotations = []
        categories = []
        last_annotation_id = 0
        last_image_id = 0

        # 处理每个输入文件
        for input_path in input_paths:
            with open(input_path, "r") as f:
                data = json.load(f)
                images.extend(data["images"])
                categories=data["categories"]

                # 更新annotation的id
                for ann in data["annotations"]:
                    ann["id"] += last_annotation_id
                    ann["image_id"] += last_image_id
                    annotations.append(ann)

                # 更新image的id
                for img in data["images"]:
                    img["id"] += last_image_id
                    last_image_id += 1

                # 更新last_annotation_id
                last_annotation_id += len(data["annotations"])

        # 创建输出JSON数据
        output_json = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        return output_json

    def _split_json(self, ann_json, train_files, test_files, val_files):
        '''
        根据划分后的数据集，分割json
        '''
        images = ann_json["images"]
        annotations = ann_json["annotations"]
        categories = ann_json["categories"]

        train_images = []
        train_annotations = []
        test_images = []
        test_annotations = []
        val_images = []
        val_annotations = []

        # 处理每个图像
        for img in images:
            if img["file_name"] in train_files:
                train_images.append(img)
            elif img["file_name"] in test_files:
                test_images.append(img)
            elif img["file_name"] in val_files:
                val_images.append(img)

        # 处理每个注释
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id in [x["id"] for x in train_images]:
                train_annotations.append(ann)
            elif img_id in [x["id"] for x in test_images]:
                test_annotations.append(ann)
            elif img_id in [x["id"] for x in val_images]:
                val_annotations.append(ann)

        # 创建输出JSON数据
        train_json = {
            "images": train_images,
            "annotations": train_annotations,
            "categories": categories
        }

        test_json = {
            "images": test_images,
            "annotations": test_annotations,
            "categories": categories
        }

        val_json = {
            "images": val_images,
            "annotations": val_annotations,
            "categories": categories
        }

        return train_json, test_json, val_json

    def _check(self, input_dir):
        if not os.path.exists(input_dir):
            raise ValueError("{} 路径错误".format(input_dir))
        # 检查classes.txt文件是否存在
        classes_path = os.path.join(input_dir, "classes.txt")
        if not os.path.exists(classes_path):
            raise ValueError("no classes.txt")

        # 检查images文件夹和其中的图像文件
        images_path = os.path.join(input_dir, "images")
        if not os.path.exists(images_path):
            raise ValueError("no images")
        # image_files = os.listdir(images_path)
        # for file_name in image_files:
        #     if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
        #         return False

        # 检查annotations文件夹和其中的注释文件
        annotations_path = os.path.join(input_dir, "annotations")
        if not os.path.exists(annotations_path):
            raise ValueError("no annotations")
        # annotation_files = os.listdir(annotations_path)
        # for file_name in annotation_files:
        #     if not (file_name.endswith(".json") or file_name.endswith(".xml") or file_name.endswith(".txt")):
        #         return False

    def check(self, dataset_path = None):
        if self.dataset_type == None:
            try:
                try:
                    self.check_det(dataset_path)
                    self.print_folder_structure(self.dataset_path)
                except:
                    self.check_cls(dataset_path)
                    self.print_folder_structure(self.dataset_path)
            except:
                print("数据集既不符合det也不符合cls")
        elif self.dataset_type == "det":
            try:
                self.check_det()
                self.print_folder_structure(self.dataset_path)
            except Exception as e:
                print(e)
        elif self.dataset_type == "cls":
            try:
                self.check_cls()
                self.print_folder_structure(self.dataset_path)
            except Exception as e:
                print(e)



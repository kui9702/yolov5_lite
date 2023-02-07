import os
import glob
import random
import shutil
import tqdm
import cv2

blue_path = r"/mnt/CCPD2019/ccpd_*/*.jpg"      # 蓝牌路径
green_path = r"/mnt/CCPD2020/ccpd_green/train/*.jpg"       # 绿牌路径
save_dir = r"/mnt/ai_train/ccpd_yolov5lite"                    # 保存位置
blue_nums = 1000                                    # 选择数量 None 为全部
green_nums = 0
tr_percent = 0.7                                    # 数据集比例
va_percent = 0.85
te_percent = 1

def select_dataset():
    global blue_path, green_path, save_dir, green_nums, blue_nums
    files = glob.glob(blue_path)
    files = random.sample(files, len(files))  #355013 need10000

    green_files = glob.glob(green_path)
    green_files = random.sample(green_files, len(green_files))  #5769 need2000
    if blue_nums is None:
        blue_nums = len(files)
    if green_nums is None:
        green_nums = len(green_files)
    total_files = files[:blue_nums]+green_files[:green_nums]
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm.tqdm(total_files):
        shutil.copyfile(i, os.path.join(save_dir, os.path.basename(i)))

def convert(size, box):
    box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def points2yolo(size, points):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    rbx = points[0] * dw
    rby = points[1] * dh
    lbx = points[2] * dw
    lby = points[3] * dh
    ltx = points[4] * dw
    lty = points[5] * dh
    rtx = points[6] * dw
    rty = points[7] * dh
    return rbx, rby, lbx, lby, ltx, lty, rtx, rty

def get_points(file1):
    points = os.path.basename(file1).split("-")[3] # 386&473_177&454_154&383_363&402 rb lb lt rt
    points = points.split("_")
    rb = points[0].split("&") # rb right_bottom
    lb = points[1].split("&") # 
    lt = points[2].split("&")
    rt = points[3].split("&")
    return int(rb[0]), int(rb[1]), int(lb[0]), int(lb[1]), int(lt[0]), int(lt[1]), int(rt[0]), int(rt[1])

def get_cor(file1):
    # 获取坐标
    try:
        x1y1x2y2 = os.path.basename(file1).split("-")[2]
        x1 = int(x1y1x2y2.split("_")[0].split("&")[0])
        y1 = int(x1y1x2y2.split("_")[0].split("&")[1])
        x2 = int(x1y1x2y2.split("_")[1].split("&")[0])
        y2 = int(x1y1x2y2.split("_")[1].split("&")[1])
        return x1, y1, x2, y2
    except:
        print(file1)
        return 0, 0, 0, 0

def copy_file(file_lists, path, str_):
    os.makedirs(os.path.join(path, "images", str_), exist_ok=True)
    for file in tqdm.tqdm(file_lists):
        if(len(os.path.basename(file)) > 25):
            os.rename(file, os.path.join(path, "images", str_, os.path.basename(file)))


def split_data():
    global save_dir
    base_dir = save_dir
    files = glob.glob(os.path.join(base_dir, "*.jpg"))
    files = random.sample(files, len(files))
    copy_file(files[:int(len(files)*tr_percent)], base_dir, "train2017")
    copy_file(files[int(len(files)*tr_percent):int(len(files)*va_percent)], base_dir, "val2017")
    copy_file(files[int(len(files)*va_percent):], base_dir, "test2017")

def get_cor_from_filename(path):
    x1, y1, x2, y2 = get_cor(path)
    rbx, rby, lbx, lby, ltx, lty, rtx, rty = get_points(path)
    if x1 > x2 or y1 > y2:
        print(path)
    image = cv2.imread(path)
    image_w = image.shape[1]
    image_h = image.shape[0]
    x, y, w, h = convert((image_w, image_h), (x1, y1, x2, y2))
    rbx, rby, lbx, lby, ltx, lty, rtx, rty = points2yolo((image_w, image_h), (rbx, rby, lbx, lby, ltx, lty, rtx, rty))
    os.makedirs(os.path.dirname(path.replace(r"images", "labels")), exist_ok=True)
    with open(path.replace(r"images", "labels").replace(r".jpg", ".txt"), "w", encoding="utf-8") as ff:
        ff.write(f"{0} {x} {y} {w} {h}\n")

def write_label():
    global save_dir
    files = glob.glob(os.path.join(save_dir, "images", "*", "*.jpg"))  # r"/mnt/e/BaiduNetdiskDownload/coco/images/*/*.jpg"
    for i in tqdm.tqdm(files):
        get_cor_from_filename(i)


if __name__ == "__main__":
    # 挑选16000张图片做训练 测试 验证集
    select_dataset()

    # 测试图片坐标
    # file1 = r"/mnt/e/BaiduNetdiskDownload/coco/0051-0_0-328&521_439&560-439&559_328&560_328&522_439&521-0_0_27_8_26_31_30-115-31.jpg"
    # image = cv2.imread(file1)
    # image = image[y1:y2, x1:x2]

    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllmnts()

    # 分离数据集
    split_data()

    write_label()
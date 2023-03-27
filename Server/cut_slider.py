import os
import cv2
import random
import math
from collections import Counter
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from histolab.tiler import GridTiler
from histolab.masks import *
from PIL import Image
from histolab.slide import Slide
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import pandas as pd
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import copy
import matplotlib.pyplot as plt
import tifffile as tiff 
from sklearn.metrics.pairwise import cosine_similarity


def read_regions(path):
    tree = ET.parse(path)
    labels = []
    region_index = []
    root = tree.getroot()
    regions = root.find("Annotations")
    for r in regions:
        print()
        labels.append(1 if r.attrib["PartOfGroup"] == "cancer" else 0)
        coordinates = r.find("Coordinates")
        c_index = []
        for c in coordinates:
            c_index.append(
                (int(c.attrib['X']), int(c.attrib['Y']))
            )
        region_index.append(c_index)
    return labels, region_index


def init_peso_WSI(
    WSI_path=None,         # \WSIs
    tile_size=128,
    level=0,
    region_path=None,     # \region
    save_patch_path=None,  # \init_data_image
    save_thumbnail_path=None,  # \WSI_thumbnail
    save_labels_path=None  # \
):
    WSIs = os.listdir(WSI_path)
    Regions = os.listdir(region_path)
    all_labels = []
    for i, (w) in enumerate(WSIs):
        r_path = w.split("_")[0]+"_"+w.split("_")[1]+'.xml'
        r_path = os.path.join(region_path, r_path)
        labels, region_index = read_regions(r_path)
        all_labels.append(labels)

        prostate_slide = Slide(os.path.join(WSI_path, w),
                               processed_path=save_patch_path)
        print("Tile WSI for iter:", i)
        print(f"Slide name: {prostate_slide.name}")
        print("Region config:", r_path)
        print(f"Levels: {prostate_slide.levels}")
        print(f"Dimensions at level 0: {prostate_slide.dimensions}")

        tuh_image = prostate_slide.thumbnail
        print("Thumbnail WSI size:", prostate_slide._thumbnail_size)
        for j, (r) in enumerate(region_index):
            idx = 4*(i)+j
            tuh_image.save(os.path.join(save_thumbnail_path, str(idx)+".png"))
            grid_tiles_extractor = GridTiler(
                tile_size=(tile_size, tile_size),
                level=level,
                check_tissue=True,  # default
                pixel_overlap=0,  # default
                suffix=".png",  # default
                region=r,
                idx=idx
            )
            grid_tiles_extractor.extract(prostate_slide)
    pd.DataFrame(all_labels).to_csv(
        os.path.join(save_labels_path, 'label.csv'))
    return all_labels


# train .....
def insert_noise_labels(labels, ratio=0.3):

    x = copy.deepcopy(labels)
    cnt = int(len(x) * ratio)
    pos = []
    i = 0
    while i < len(x) - 1:
        if x[i] != x[i + 1]:
            pos.append(i)
            i += 1
        i += 1
    random.shuffle(pos)

    for i in range(cnt // 2):
        x[pos[i]], x[pos[i] + 1] = x[pos[i] + 1], x[pos[i]]
    return x


def overlap(box1, box2):
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return False
    else:
        return True


def split_region(image, height=25, width=25, cnt=4, background_rate=0.05, balance_rate=0.46):
    x, y, _ = image.shape
    pos = []
    for dx in range(x):
        for dy in range(y):
            if dx + height > x or dy + width > y:
                continue
            cnt0 = np.sum(image[dx:dx + height, dy:dy + width, 0] == 0)
            cnt1 = np.sum(image[dx:dx + height, dy:dy + width, 0] == 1)
            cnt2 = np.sum(image[dx:dx + height, dy:dy + width, 0] == 2)
            cntall = height * width
            if min(cnt1, cnt2) / (cnt1 + cnt2 + 1) < balance_rate or cnt0 / cntall > background_rate:
                continue
            flag = 1
            for (x1, y1, x2, y2, _) in pos:
                if overlap((x1, y1, x2, y2), (dx, dy, dx + height - 1, dy + width - 1)):
                    flag = 0
                    break
            if flag:
                pos.append((dx, dy, dx + height - 1, dy + width -
                           1, min(cnt1, cnt2) / (cnt1 + cnt2)))
    pos.sort(key=lambda x: -x[4])

    return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in pos[:cnt]], [(r) for (_, _, _, _, r) in pos[:cnt]]


def init_peso_WSI_train(
    wsi_path,
    mask_path,
    save_path,
    thumbnails_path,
    tile_size: int = 128,
    region_h: int = 30,
    region_w: int = 30
):
    wsis = os.listdir(wsi_path)
    region_num = 0
    true_label = []
    # start tile
    for w in tqdm(wsis):
        slide = OpenSlide(os.path.join(wsi_path, w))
        masks = OpenSlide(os.path.join(mask_path, w[:-4]+"_training_mask.tif"))
        # print("WSI dimensions in each level:", slide.level_dimensions)
        # print("WSI level downsamples:", slide.level_downsamples)
        # get masks array
        masks_arr = np.array(masks.read_region(
            location=(0, 0), level=7, size=slide.level_dimensions[-1]))
        # get regions
        regions, _ = split_region(image=masks_arr)
        # print("We select %d regions in this WSI" % len(regions))
        # tile regions
        for x1, y1, x2, y2 in regions:
            # get and save thumbnails
            thu = np.array(slide.read_region(location=(tile_size*y1, tile_size*x1),
                                             level=0, size=(region_h*tile_size, region_w*tile_size)).resize((tile_size, tile_size)))
            Image.fromarray(thu).save(os.path.join(
                thumbnails_path, str(region_num)+".png"))
            # in this region
            for i in range(x1, x2+1):
                for j in range(y1, y2+1):
                    # label
                    label = masks_arr[i, j, :][0]
                    if label == 1 or label == 2:
                        s = np.array(slide.read_region(
                            location=(tile_size*j, tile_size*i), level=0, size=(tile_size, tile_size)))
                        # make dir
                        sp = os.path.join(save_path, "image_"+str(region_num))
                        if not os.path.exists(sp):
                            os.makedirs(sp)
                        # save patch
                        Image.fromarray(s).save(os.path.join(
                            sp, "x_%d_y_%d_%d.png" % (i, j, label-1)))
                        # get label
                        true_label.append(label-1)  # label
            # region number +1
            region_num += 1
    return region_num


def init_peso_WSI_train_1024(
    wsi_path,
    mask_path,
    save_path,
    thumbnails_path,
    ts: int = 128,
    level: int = 3,  # level+1->ts/2; level-1->ts*2
    tile_size: int = 1024,
    input_size: int = 224,
    background_rate: float = 0.2,
    cancer_act_rate: float = 0.25,
    number: int = 20,
):
    min_i = 10000
    min_j = 10000
    true_label = []
    wsis = os.listdir(wsi_path)
    balance_l = np.zeros(number)
    pbar = tqdm(total=number, desc="Count", unit="it")
    for idx, (w) in enumerate(wsis[len(wsis)-number:]):
        slide = OpenSlide(os.path.join(wsi_path, w))
        masks = OpenSlide(os.path.join(mask_path, w[:-4]+"_training_mask.tif"))
        masks_arr = np.array(masks.read_region(
            location=(0, 0), level=level, size=slide.level_dimensions[level]))
        tile_label = np.zeros((masks_arr.shape[0]//ts, masks_arr.shape[1]//ts))
        # get and save thumbnails
        thu = np.array(slide.read_region(
            location=(0, 0), level=level, size=slide.level_dimensions[level]))
        Image.fromarray(thu).resize((input_size, input_size)).save(os.path.join(
            thumbnails_path, str(idx)+".png"))
        for i in range(tile_label.shape[0]):
            for j in range(tile_label.shape[1]):
                mask_arr_ = masks_arr[i*ts:(i+1)*ts, j*ts:(j+1)*ts, 0]
                if np.sum(mask_arr_ == 0)/(ts*ts) < background_rate:
                    if np.sum(mask_arr_ == 2)/np.sum(mask_arr_ == 1) >= cancer_act_rate:
                        tile_label[i, j] = 2
                        label = 1
                    else:
                        tile_label[i, j] = 1
                        label = 0
                    s = np.array(slide.read_region(
                        location=(tile_size*j, tile_size*i), level=0, size=(tile_size, tile_size)))
                    # make dir
                    sp = os.path.join(save_path, "image_"+str(idx))
                    if not os.path.exists(sp):
                        os.makedirs(sp)
                    # save patch
                    Image.fromarray(s).resize((input_size, input_size)).save(os.path.join(
                        sp, "x_%d_y_%d_%d.png" % (i, j, label)))
                    # save label
                    true_label.append(label)
                    # save min i,j
                    min_i = min(min_i, i)
                    min_j = min(min_j, j)
        # change i,j
        for img in os.listdir(sp):
            s = img[:-4].split("_")
            i = int(s[1])
            j = int(s[3])
            l = int(s[4])
            Image.open(os.path.join(sp, img)).save(os.path.join(
                sp, "x_%d_y_%d_%d.png" % (i-min_i, j-min_j, l)))
            os.remove(os.path.join(sp, img))
        # caculate sample balance
        cnt0 = np.sum(tile_label == 1)
        cnt1 = np.sum(tile_label == 2)

        balance_l[idx] = cnt1/(cnt0+cnt1)
        #
        min_i = 10000
        min_j = 10000
        pbar.update(1)
    print("The cancer sample rate is %.3f" % np.mean(balance_l))

    return true_label


def npy_data_init(
        img_path,  # /init_data_image
        save_path,  # /init_data
        label_rate=0.5,
        test_rate=0.2,
        size=224,
        img_num=0,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # imglen是wsi图的个数
    train_dataset = []
    labeled_set = []
    unlabeled_set = []
    test_dataset = []
    files = []

    transformations = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    global_index = 0
    for imgid in range(img_num):
        files_path = []
        path = os.path.join(img_path, 'image_'+str(imgid))

        for file_name in os.listdir(path):
            files_path.append(file_name)

        patch_num = len(files_path)
        #
        rand_init = torch.randperm(patch_num)
        labeled_index = rand_init[:int(label_rate*patch_num)]
        unlabeled_index = rand_init[int(
            label_rate*patch_num): int((1-test_rate)*patch_num)]

        for patch_index in rand_init[:int((1-test_rate)*patch_num)]:
            files.append((imgid, patch_index, files_path[patch_index]))
            image = Image.open(os.path.join(path, files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(files_path[patch_index][-5])
            train_dataset.append(
                (img, label, patch_index, imgid, global_index))
            if patch_index in labeled_index:
                labeled_set.append(
                    (img, label, patch_index, imgid, global_index))

            elif patch_index in unlabeled_index:
                unlabeled_set.append(
                    (img, label, patch_index, imgid, global_index))

            global_index += 1

        for patch_index in rand_init[int((1-test_rate)*patch_num):]:
            image = Image.open(os.path.join(path, files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(files_path[patch_index][-5])
            test_dataset.append((img, label, patch_index, imgid, 0))

    train_dataset = np.array(train_dataset)
    labeled_set = np.array(labeled_set)
    unlabeled_set = np.array(unlabeled_set)
    test_dataset = np.array(test_dataset)
    print(len(train_dataset), len(labeled_set), len(
        unlabeled_set), len(test_dataset), len(files))

    #
    np.save(os.path.join(save_path, "patch_file.npy"), np.array(files))

    np.save(os.path.join(save_path, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(save_path, "init_labeled_set.npy"), labeled_set)
    np.save(os.path.join(save_path, "init_unlabeled_set.npy"), unlabeled_set)
    np.save(os.path.join(save_path, "test_dataset.npy"), test_dataset)


def compress_images(
    img_path,  # /init_data_image
    save_path,  # /init_data
    compress_size=64
):
    img_paths = os.listdir(img_path)
    for p in tqdm(img_paths):
        pp = os.path.join(img_path, p)
        imgs = os.listdir(pp)
        for i in imgs:
            sp = os.path.join(save_path, p)
            if not os.path.exists(sp):
                os.makedirs(sp)
            Image.open(os.path.join(pp, i)).resize(
                (compress_size, compress_size)).save(os.path.join(sp, i))
    return


#
def init_peso_WSI_train_percent(
    wsi_path,
    mask_path,
    save_path,
    thumbnails_path,
    ts: int = 128,
    level: int = 3,  # level+1->ts/2; level-1->ts*2
    tile_size: int = 1024,
    input_size: int = 224,
    background_rate: float = 0.998,
    interval:int=0.05,
    cancer_act_rate: list = [0.05, 0.15, 0.45, 0.9],  # no cancer ,cancer,deep cancer
    number: int = 30,
):
    min_i = 10000
    min_j = 10000
    true_label = []
    wsis = os.listdir(wsi_path)
    balance=np.zeros(len(cancer_act_rate))
    pbar = tqdm(total=number, desc="Count", unit="it")
    print(wsis)
    for idx, (w) in enumerate(wsis[:number]):
        slide = OpenSlide(os.path.join(wsi_path, w))
        print("\nID:%d slide name:%s"%(idx,w))
        masks = OpenSlide(os.path.join(mask_path, w[:-4]+"_training_mask.tif"))
        masks_arr = np.array(masks.read_region(
            location=(0, 0), level=level, size=slide.level_dimensions[level]))
        tile_label = np.zeros((masks_arr.shape[0]//ts, masks_arr.shape[1]//ts))-1
        # get and save thumbnails
        if 0:
            thu = np.array(slide.read_region(
                location=(0, 0), level=level, size=slide.level_dimensions[level]))
            Image.fromarray(thu).resize((input_size,input_size)).save(os.path.join(
                thumbnails_path, str(idx)+".png"))
        for i in range(tile_label.shape[0]):
            for j in range(tile_label.shape[1]):
                mask_arr_ = masks_arr[i*ts:(i+1)*ts, j*ts:(j+1)*ts, 0]
                if np.sum(mask_arr_ == 0)/(ts*ts) < background_rate:
                    if np.sum(mask_arr_ == 0)/(ts*ts) < 0.2:
                        rate = np.sum(mask_arr_ == 2)/np.sum(mask_arr_ ==1)
                        if rate > cancer_act_rate[0] and rate < cancer_act_rate[1]:
                            label = 1  # no cancer
                        elif rate > cancer_act_rate[1]+interval and rate < cancer_act_rate[2]:
                            label = 2  # cancer
                        elif rate > cancer_act_rate[2]+interval and rate < cancer_act_rate[3]:
                            label = 3  # more cancer
                        else :
                            label = 0 # 无效label
                    else:
                        label=0
                    #
                    tile_label[i, j] = label
                    s = np.array(slide.read_region(
                        location=(tile_size*j, tile_size*i), level=0, size=(tile_size, tile_size)))
                    # make dir
                    sp = os.path.join(save_path, "image_"+str(idx))
                    if not os.path.exists(sp):
                        os.makedirs(sp)
                    # save patch
                    if 1:
                        Image.fromarray(s).save(os.path.join(
                            sp, "x_%d_y_%d.png" % (i, j)))
                    else:
                        Image.fromarray(s).resize((input_size,input_size)).save(os.path.join(
                        sp, "x_%d_y_%d_%d.png" % (i, j, label)))
                    # save label
                    true_label.append(label)
                    # save min i,j
                    min_i = min(min_i, i)
                    min_j = min(min_j, j)
        # # change i,j
        for img in os.listdir(sp):
            s=img[:-4].split("_")
            i=int(s[1])
            j=int(s[3])
            
            if 1:
                Image.open(os.path.join(sp,img)).save(os.path.join(sp,"x_%d_y_%d.png" % (i-min_i+1, j-min_j+1)))
            else:
                l=int(s[4])
                Image.open(os.path.join(sp,img)).save(os.path.join(sp,"x_%d_y_%d_%d.png" % (i-min_i+1, j-min_j+1, l)))
            os.remove(os.path.join(sp,img))
        # caculate sample balance
        # cnt=[np.sum(tile_label == 0),np.sum(tile_label == 1),np.sum(tile_label == 2),np.sum(tile_label == 3)]
        # s = sum(cnt)
        # for i in range(balance.shape[0]):
        #     balance[i]+=cnt[i]
        # print("The cancer sample number is:", cnt)
        #
        min_i = 10000
        min_j = 10000
        pbar.update(1)
    # print("The ALL cancer sample number is:", balance)
    return true_label


def npy_data_init_percent(
        img_path,  # /init_data_image
        save_path,  # /init_data
        label_rate=0.5,
        test_rate=0.2,
        size=224,
        img_num=0,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # imglen是wsi图的个数
    train_dataset = []
    labeled_set = []
    unlabeled_set = []
    test_dataset = []
    files = []
    ts_files=[]


    transformations = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    global_index = 0
    for imgid in tqdm(range(25,img_num)):
        files_path = []
        path = os.path.join(img_path, 'image_'+str(imgid))

        for file_name in os.listdir(path):
            lb=int(file_name.split('.')[0][-1]) 
            if lb!=0:
                files_path.append(file_name)

        patch_num = len(files_path)
        #
        rand_init = torch.randperm(patch_num)
        labeled_index = rand_init[:int(label_rate*patch_num)]
        unlabeled_index = rand_init[int(
            label_rate*patch_num): int((1-test_rate)*patch_num)]
        
        
        # labeled and unlabeled
        for patch_index in rand_init[:int((1-test_rate)*patch_num)]:
            files.append((imgid, patch_index, files_path[patch_index]))
            image = Image.open(os.path.join(path, files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(files_path[patch_index][-5])
            train_dataset.append(
                (img, label, patch_index, imgid, global_index))
            if patch_index in labeled_index:
                labeled_set.append(
                    (img, label, patch_index, imgid, global_index))

            elif patch_index in unlabeled_index:
                unlabeled_set.append(
                    (img, label, patch_index, imgid, global_index))

            global_index += 1
        
        
        # test
        for patch_index in rand_init[int((1-test_rate)*patch_num):]:
            ts_files.append((imgid, patch_index, files_path[patch_index]))
            image = Image.open(os.path.join(path, files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(files_path[patch_index][-5])
            test_dataset.append((img, label, patch_index, imgid, 0))

    train_dataset = np.array(train_dataset)
    labeled_set = np.array(labeled_set)
    unlabeled_set = np.array(unlabeled_set)
    test_dataset = np.array(test_dataset)
    
    print(len(train_dataset), len(labeled_set), len(
        unlabeled_set), len(test_dataset), len(files))

    #
    np.save(os.path.join(save_path, "patch_file.npy"), np.array(files))
    np.save(os.path.join(save_path, "ts_patch_file.npy"), np.array(ts_files))

    np.save(os.path.join(save_path, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(save_path, "init_labeled_set.npy"), labeled_set)
    np.save(os.path.join(save_path, "init_unlabeled_set.npy"), unlabeled_set)
    np.save(os.path.join(save_path, "test_dataset.npy"), test_dataset)


def npy_data_init_bk(
        img_path,  # /init_data_image
        save_path,  # /init_data
        size=224,
        img_num=0,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # imglen是wsi图的个数
    bk_files=[]
    bk_data=[]

    transformations = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    global_index = 0
    for imgid in tqdm(range(img_num)):
        bk_files_path=[]
        path = os.path.join(img_path, 'image_'+str(imgid))

        for file_name in os.listdir(path):
            lb=int(file_name.split('.')[0][-1]) 
            if lb==0:
                bk_files_path.append(file_name)
        print(imgid,len(bk_files_path))
        # bk
        for patch_index in range(len(bk_files_path)):
            bk_files.append((imgid, patch_index, bk_files_path[patch_index]))
            image = Image.open(os.path.join(path, bk_files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(bk_files_path[patch_index][-5])
            bk_data.append(
                (img, label, patch_index, imgid, global_index))
    
    bk_data = np.array(bk_data)
    bk_files = np.array(bk_files)

    print(len(bk_data),len(bk_files))
    np.save(os.path.join(save_path, "bk_dataset.npy"), bk_data)
    np.save(os.path.join(save_path, "bk_patch_file.npy"), bk_files)


def mask_transform(
    wsi_path,
    mask_path,
    mask_save_path,
    levels:list=[5],
    number: int = 40,
):
    colors=[
        [255,255,255,255],
        [137,212,94,255],
        [240,65,69,255],
    ]
    wsis = os.listdir(wsi_path)
    for l in levels:
        sp=os.path.join(mask_save_path, "level_"+str(l))
        if not os.path.exists(sp):
            os.makedirs(sp)
        for idx, (w) in enumerate(wsis[:number]):
            slide = OpenSlide(os.path.join(wsi_path, w))
            masks = OpenSlide(os.path.join(mask_path, w[:-4]+"_training_mask.tif"))
            masks_arr = np.array(masks.read_region(
                location=(0, 0), level=l, size=masks.level_dimensions[l]))
            slide_arr=np.array(slide.read_region(
                location=(0, 0), level=l, size=slide.level_dimensions[l]))
            for i in range(masks_arr.shape[0]):
                for j in range(masks_arr.shape[1]):
                    masks_arr[i,j,:]=colors[masks_arr[i,j,0]]

            Image.fromarray(masks_arr).save(os.path.join(sp, "mask_"+str(idx)+".tif"))
            Image.fromarray(slide_arr).save(os.path.join(sp,"slide_"+ str(idx)+".tif"))


def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T



# 计算图片的余弦距离
def image_similarity_vectors(image1, image2):
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image:
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = np.dot(a / a_norm, b / b_norm)
    return res*10000-int(res*10000)



def init_hubmap_WSI(
    wsi_path,
    mask_path,
    save_path,
    transform_path,
    thumbnails_path,
    tile_size:int=200,
    cancer_act_rate:float=0.2, 
    interval:int=0.1,
    number=200,
):
    plt.style.use("Solarize_Light2")     
    wsis = os.listdir(wsi_path)
    df = pd.read_csv(
        os.path.join(mask_path, "train.csv")
    )
    back=np.array(Image.open(r"D:\DATASETS\hubmap\back.png"))
    tile_label=[]
    pbar = tqdm(total=number, desc="Count", unit="it")
    for idx, (w) in enumerate(wsis[:number]):
        slide=np.array(tiff.imread(os.path.join(wsi_path, w))) 
        slide_id=int(w.split(".")[0])
        masks=rle2mask(df[df["id"]==slide_id]["rle"].iloc[-1], (slide.shape[1], slide.shape[0]))
        #
        Image.fromarray(slide).resize((224, 224)).save(os.path.join(
            thumbnails_path, str(idx)+".png"))
        # 
        plt.figure(figsize=(10,10))
        plt.imshow(slide)
        plt.imshow(masks, alpha=0.5)
        plt.axis("off")
        plt.savefig(os.path.join(transform_path,str(idx)+".png"))
        plt.close()
        for i in range(masks.shape[0]//tile_size):
            for j in range(masks.shape[1]//tile_size):
                mask_arr_ = masks[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                rate = np.sum(mask_arr_ == 1)/(tile_size**2)
                if rate >=0 and rate <=cancer_act_rate-interval:
                    label = 1  # no cancer
                elif rate >= cancer_act_rate:
                    label = 2  # cancer
                else :
                    label = 0 # 无效label
                s=slide[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size,:]
                print(s[0,0,:])
                # if image_similarity_vectors(back,s)>0.9:
                #     label = 0
                # make dir
                sp = os.path.join(save_path, "image_"+str(idx))
                if not os.path.exists(sp):
                    os.makedirs(sp)
                # save patch
                # Image.fromarray(s).save(os.path.join(
                #     sp, "x_%d_y_%d_%d.png" % (i, j, label)))
                
                tile_label.append(label)
        pbar.update(1)
    tile_label=np.array(tile_label)
    print(np.sum(tile_label==0),np.sum(tile_label==1),np.sum(tile_label==2))

    return



if __name__ == '__main__':
    # peso
    wsi_path = r"F:\data\peso_train_wsi"
    mask_path = r"F:\data\peso_train_masks"
    mask_save_path=r"D:\DATASETS\peso\train\mask_transform"
    thumbnails_path = r"D:\DATASETS\peso\train\thumbnails"
    save_img_path = r"D:\DATASETS\peso\train\init_data_image"
    save_init_path = r"D:\DATASETS\peso\train\init_data"
    compress_path = r"D:\DATASETS\peso\train\compress"
    #
    img_num = 30
    init_peso_WSI_train_percent(
        wsi_path=wsi_path,
        mask_path=mask_path,
        save_path=r"F:\data\visualization\peso\init_data_image1024",
        thumbnails_path=thumbnails_path,
        number=img_num,
    )
    
    # mask_transform(
    #     wsi_path=wsi_path,
    #     mask_path=mask_path,
    #     mask_save_path=mask_save_path,
    # )    
    # init_peso_WSI_train_percent(
    #     wsi_path=wsi_path,
    #     mask_path=mask_path,
    #     save_path=save_img_path,
    #     thumbnails_path=thumbnails_path,
    #     number=img_num,
    # )
    
    # npy_data_init_percent(
    #     img_path=save_img_path,
    #     save_path=save_init_path,
    #     img_num=img_num
    # )
    # npy_data_init_bk(
    #     img_path=save_img_path,
    #     save_path=save_init_path,
    #     img_num=img_num
    # )

    # compress_images(
    #     img_path=save_img_path,
    #     save_path=compress_path,
    #     compress_size=60,
    # )
    

    # hubmap
    # wsi_path = r"D:\DATASETS\hubmap\train_images"
    # mask_path = r"D:\DATASETS\hubmap"
    # thumbnails_path = r"D:\DATASETS\hubmap\thumbnails"
    # transform_path= r"D:\DATASETS\hubmap\mask_transform"
    # save_img_path = r"D:\DATASETS\hubmap\init_data_image"
    # save_init_path = r"D:\DATASETS\hubmap\init_data"
    # compress_path = r"D:\DATASETS\hubmap\compress"

    # hubmap_number=100

    
    # init_hubmap_WSI(
    #     wsi_path=wsi_path,
    #     mask_path=mask_path,
    #     save_path=save_img_path,
    #     thumbnails_path=thumbnails_path,
    #     transform_path=transform_path,
    #     number=hubmap_number
    # )
    # npy_data_init_percent(
    #     img_path=save_img_path,
    #     save_path=save_init_path,
    #     size=200,
    #     img_num=hubmap_number
    # )
    # npy_data_init_bk(
    #     img_path=save_img_path,
    #     save_path=save_init_path,
    #     size=200,
    #     img_num=hubmap_number
    # )


    pass

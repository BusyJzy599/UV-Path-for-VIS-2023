import os
import random
import math
from collections import Counter
from tqdm import tqdm
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import pandas as pd
import copy
import matplotlib.pyplot as plt
import tifffile as tiff 
from sklearn.metrics.pairwise import cosine_similarity



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
                            label = 0 # 
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

if __name__ == '__main__':
    # peso
    wsi_path = r""
    mask_path = r""
    mask_save_path=r""
    thumbnails_path = r""
    save_img_path = r""
    save_init_path = r""
    compress_path = r""
    #
    img_num = 30
    mask_transform(
        wsi_path=wsi_path,
        mask_path=mask_path,
        mask_save_path=mask_save_path,
    )    
    init_peso_WSI_train_percent(
        wsi_path=wsi_path,
        mask_path=mask_path,
        save_path=save_img_path,
        thumbnails_path=thumbnails_path,
        number=img_num,
    )
    npy_data_init_percent(
        img_path=save_img_path,
        save_path=save_init_path,
        img_num=img_num
    )
    npy_data_init_bk(
        img_path=save_img_path,
        save_path=save_init_path,
        img_num=img_num
    )

    compress_images(
        img_path=save_img_path,
        save_path=compress_path,
        compress_size=60,
    )
    
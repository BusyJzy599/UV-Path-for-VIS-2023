
from sklearn.model_selection import KFold
import pandas as pd
from utils.additional import *
from utils.augmentation import *
from utils.model_utils import *
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from random import shuffle




LABELS = 'D:/DATASETS/hubmap/1000_pixel_images/train.csv'
is_amp = True

def make_fold1000(organ,fold=0,label_ratio=0.6,noise_ratio=0.3,seed=42):
    df = pd.read_csv(LABELS)
    df=df[df.organ==organ]

    num_fold = 5
    skf = KFold(n_splits=num_fold, shuffle=True, random_state=seed)

    df.loc[:, 'fold'] = -1
    for f, (t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
        df.iloc[v_idx, -1] = f


    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)
    
    labeled_df=train_df.iloc[:int(label_ratio*len(train_df))]
    unlabeled_df=train_df.iloc[int(label_ratio*len(train_df)):]
    # get noise
    x = [x for x in range(len(labeled_df))]
    shuffle(x)
    noise_df=labeled_df.iloc[x[:int(noise_ratio*len(labeled_df))]]
    noise_masks=noisify_seg(noise_df)
    
    return train_df,labeled_df,unlabeled_df, valid_df,noise_masks

def make_fold(fold=0):
    df = pd.read_csv(LABELS)

    num_fold = 5
    skf = KFold(n_splits=num_fold, shuffle=True, random_state=42)

    df.loc[:, 'fold'] = -1
    for f, (t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
        df.iloc[v_idx, -1] = f

    # check
    if 0:
        for f in range(num_fold):
            train_df = df[df.fold != f].reset_index(drop=True)
            valid_df = df[df.fold == f].reset_index(drop=True)

            print('fold %d' % f)
            t = train_df.organ.value_counts().to_dict()
            v = valid_df.organ.value_counts().to_dict()
            for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
                print('%32s %3d (%0.3f)  %3d (%0.3f)' % (k, t.get(k, 0), t.get(
                    k, 0)/len(train_df), v.get(k, 0), v.get(k, 0)/len(valid_df)))

            print('')
            zz = 0

    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)
    return train_df, valid_df


def compute_dice_score(probability, mask):
    N = len(probability)
    p = probability.reshape(N, -1)
    t = mask.reshape(N, -1)

    p = p > 0.5
    t = t > 0.5

    uion = p.sum(-1) + t.sum(-1)
    overlap = (p*t).sum(-1)
    dice = 2*overlap/(uion+0.0001)
    return dice


def compute_iou_score(probability, mask):
    smooth = 1e-5
    N = len(probability)
    p = probability.reshape(N, -1)
    t = mask.reshape(N, -1)

    p = p > 0.5
    t = t > 0.5
    intersection = (p & t).sum()
    union = (p | t).sum()

    return (intersection + smooth) / (union + smooth)


def save_seg_result(ids,probability, mask,save_path):
    for idx,p,m in zip(ids,probability,mask):
        c,h,w=p.shape
        p=p.reshape(h,w)>0.5
        m = m.reshape(h,w) > 0.5
        sp=np.zeros((h,w,3))
        sm=np.zeros((h,w,3))
        for i in range(h):
            for j in range(w):
                sp[i,j]=[1,1,1] if p[i,j] else [0,0,0]
                sm[i,j]=[1,1,1] if m[i,j] else [0,0,0]
        Image.fromarray(np.uint8(sp*255)).save(os.path.join(save_path,str(idx)+"_prob.png"))
        Image.fromarray(np.uint8(sm*255)).save(os.path.join(save_path,str(idx)+"_mask.png"))         
            
        
        
        
        

def validate1000(net,valid_loader,config,iteration):
    valid_num = 0
    valid_probability = []
    valid_ids = []
    valid_mask = []
    valid_loss = 0

    net = net.eval()
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):

        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled=is_amp):

                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['mask'] = batch['mask'].cuda()

                output,_,_ = net(batch)
                loss0 = output['bce_loss'].mean()
        valid_ids.extend(batch['id'])
        valid_probability.append(output['probability'].data.cpu().numpy())
        valid_mask.append(batch['mask'].data.cpu().numpy())
        valid_num += batch_size
        valid_loss += batch_size*loss0.item()
        
    assert (valid_num == len(valid_loader.dataset))

    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)
   
    loss = valid_loss/valid_num
    dice = compute_dice_score(probability, mask)
    dice = dice.mean()
    iou = compute_iou_score(probability, mask)
    iou = iou.mean()
    if (iteration+1)%5==0:
        save_seg_result(valid_ids,probability, mask,os.path.join(config['OUT_DIR'],'valid'))
    return [loss, dice, iou]

def validate(net, valid_loader):

    valid_num = 0
    valid_probability = []
    valid_mask = []
    valid_loss = 0

    net = net.eval()
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):

        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled=is_amp):

                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['mask'] = batch['mask'].cuda()
                batch['organ'] = batch['organ'].cuda()

                output = net(batch)
                loss0 = output['bce_loss'].mean()

        valid_probability.append(output['probability'].data.cpu().numpy())
        valid_mask.append(batch['mask'].data.cpu().numpy())
        valid_num += batch_size
        valid_loss += batch_size*loss0.item()

        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset),
              (time.time() - start_timer)), end='', flush=True)

    assert (valid_num == len(valid_loader.dataset))

    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)

    loss = valid_loss/valid_num

    dice = compute_dice_score(probability, mask)
    dice = dice.mean()
    iou = compute_iou_score(probability, mask)
    iou = iou.mean()

    return [dice, loss,  iou, 0]


def binary_to_rgb(t):
    c,H,W=t.shape
    t=t.reshape((H,W))
    r=np.zeros((H,W,3))
    for i in range(H):
        for j in range(W):
            r[i,j]=[ t[i,j]*255,t[i,j]*255,t[i,j]*255 ]  
    return r


    
        
            
        

        
   
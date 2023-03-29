from __future__ import absolute_import
from torch.utils.data import  Dataset
import os
import cv2
from utils.additional import *
import torch
import pandas as pd



class HubmapDataset1000(Dataset):
    def __init__(self,config,df,organ,augment=None,noise_masks=None):
        self.config=config
        self.df = df
        self.organ=organ
        self.augment = augment
        self.noise_masks = noise_masks
        self.length = len(self.df)
        ids = self.df.id.astype(str).values
        self.fnames = [fname for fname in os.listdir(os.path.join(self.config['TRAIN'],organ)) if fname.split('.')[0] in ids]
        self.organ_to_label = {'kidney' : 0,
                               'prostate' : 1,
                               'largeintestine' : 2,
                               'spleen' : 3,
                               'lung' : 4,
                               }
    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        fname = self.fnames[index]
        idx=fname.split('.')[0]
        d = self.df.iloc[index]
        organ = self.organ_to_label[d.organ]
        fp=os.path.join(self.config['TRAIN'],self.organ)
        mfp=os.path.join(self.config['MASKS'],self.organ)
        
        image = cv2.cvtColor(cv2.imread(os.path.join(fp,fname)), cv2.COLOR_BGR2RGB)

        if self.noise_masks == None or len(self.noise_masks)==0:
            mask = cv2.imread(os.path.join(mfp,fname),cv2.IMREAD_GRAYSCALE)
        else:
            for id,org,msk in self.noise_masks:
                if id==idx:
                    mask = msk # insert noise
                else:
                    mask = cv2.imread(os.path.join(mfp,fname),cv2.IMREAD_GRAYSCALE)
       
        image = image.astype(np.float32)/255
        mask  = mask.astype(np.float32)
        
        r ={}
        r['index']= index
        r['id'] = idx
        r['organ']=torch.tensor([organ], dtype=torch.long)
        r['image'] = image_to_tensor(image)
        r['mask' ] = mask_to_tensor(mask)

        return r




LABELS = ''
image_size = 768
class HubmapDataset(Dataset):
    def __init__(self, df, augment=None):

        self.df = df
        self.augment = augment
        self.length = len(self.df)
        ids = pd.read_csv(LABELS).id.astype(str).values
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
        self.organ_to_label = {'kidney' : 0,
                               'prostate' : 1,
                               'largeintestine' : 2,
                               'spleen' : 3,
                               'lung' : 4}

    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)

        d = self.df.organ.value_counts().to_dict()
        for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
            string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        fname = self.fnames[index]
        d = self.df.iloc[index]
        organ = self.organ_to_label[d.organ]

        image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        
        image = image.astype(np.float32)/255
        mask  = mask.astype(np.float32)/255

        s = d.pixel_size/0.4 * (image_size/3000)
        image = cv2.resize(image,dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)

        if self.augment is not None:
            image, mask = self.augment(image, mask, organ)


        r ={}
        r['index']= index
        r['id'] = fname
        r['organ'] = torch.tensor([organ], dtype=torch.long)
        r['image'] = image_to_tensor(image)
        r['mask' ] = mask_to_tensor(mask>0.0) ###
       
        return r
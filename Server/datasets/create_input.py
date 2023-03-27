import os
from utils.model_utils import *
from utils.segValidation import *
from .hubmap_dataset import *
import numpy as np
from utils.model_utils import *
from utils.make_dir import *

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from torch.utils.data import SequentialSampler

class CreateDataInput(object):
    def __init__(self):
        pass


class PesoTrain(CreateDataInput):
    labeled_set = []
    unlabeled_set = []
    test_dataset = []
    train_dataset = []
    true_labels = []
    files = []
    bk_dataset=[]
    noise_idx=[]
    WSI_Data = {
        'img_id': [],
        'patch_num': [],
        'grades0': [],
        'grades1': [],
        'grades2': [],
        'o2us': [],
        'fines': []
    }
    epoch_Data = {
        'epoch': [],
        'acc': [],
        'auc': [],
        'labeled': [],
        'unlabeled': [],
        'noise_in_labeled': [],
        'infor_in_unlabled': []
    }
    sample_data = {
        'patch_id': [],
        'img_id': [],
        'is_labeled': [],
        'scatter_x': [],
        'scatter_y': [],
        'grade': [],
        'o2u': [],
        'fine': [],
        'class': [],
        'heat_score': [],
        'grades_num': [],
        'o2us_num': [],
        'fines_num': [],
        'file_name': [],
        'CAM_file_name': [],
        'noise': [],
        'kmeans_label':[]
    }
    bk_data={
        'patch_id': [],
        'img_id': [],
        'class': [], # predict
        'file_name': [],
        'kmeans_label':[] # predict
        
    }
    def __init__(self, config, logger):
        super(PesoTrain, self).__init__()
        self.path = config["dataset"]["path"]
        self.config = config
        self.logger = logger

    def init_data(self):
        self.logger.info("init datasets...")
        ls = np.load(os.path.join(
            self.path, "init_labeled_set.npy"), allow_pickle=True)
        uls = np.load(os.path.join(
            self.path, "init_unlabeled_set.npy"), allow_pickle=True)
        tes = np.load(os.path.join(
            self.path, "test_dataset.npy"), allow_pickle=True)
        trs = np.load(os.path.join(
            self.path, "train_dataset.npy"), allow_pickle=True)
        bkd=np.load(os.path.join(
            self.path, "bk_dataset.npy"), allow_pickle=True)
        bkdf=np.concatenate((
                np.load(os.path.join(self.path, "bk_patch_file.npy"), allow_pickle=True),
                np.load(os.path.join(self.path, "ts_patch_file.npy"), allow_pickle=True)
                ))
        PesoTrain.files = np.load(os.path.join(
            self.path, "patch_file.npy"), allow_pickle=True)
        PesoTrain.labeled_set = ls.tolist()
        PesoTrain.unlabeled_set = uls.tolist()
        PesoTrain.test_dataset = tes.tolist()
        PesoTrain.train_dataset = trs.tolist()
        PesoTrain.bk_dataset=np.concatenate((bkd,tes)).tolist()
        print(len(PesoTrain.bk_dataset),len(bkdf))
     
        for i in range(len(PesoTrain.train_dataset)):
            PesoTrain.true_labels.append(PesoTrain.train_dataset[i][1])
        train_labels = np.asarray([[PesoTrain.true_labels[i]]
                                   for i in range(len(PesoTrain.true_labels))], dtype=int)
        # tl=np.copy(train_labels)
        # self.logger.info("We get %d cancer samples and %d no cancer samples "%(np.sum(tl==1),tl.sum(tl==0)))
        self.logger.info("Insert noise labels")
        train_noisy_labels, actual_noise_rate = noisify(nb_classes=self.config['dataset']['num_class'],
                                                        train_labels=train_labels,
                                                        noise_type=self.config['noise_method'],
                                                        noise_rate=self.config["noise_rate"],
                                                        random_state=self.config['seed'])
        #
        if self.config['noise_range']=='all': # all : add noise in unlabeled samples
            for i, (x, label, index, img_idx, global_idx) in enumerate(PesoTrain.unlabeled_set):
                tnl=train_noisy_labels[global_idx][0]
                if tnl!=PesoTrain.unlabeled_set[i][1]:
                    PesoTrain.unlabeled_set[i][1] = tnl
                    PesoTrain.noise_idx.append(global_idx) # get noise index 
        for i, (x, label, index, img_idx, global_idx) in enumerate(PesoTrain.labeled_set):
            tnl=train_noisy_labels[global_idx][0]
            if tnl!=PesoTrain.labeled_set[i][1]:
                PesoTrain.labeled_set[i][1] = tnl
                PesoTrain.noise_idx.append(global_idx) # get noise index 

        self.logger.info("We got %d noise samples which accounts for %.2f percent of train dataset."%(len(PesoTrain.noise_idx),len(PesoTrain.noise_idx)/train_labels.size))

        # init
        img_idxs = []
        for x, label, index, img_idx, global_idx in PesoTrain.train_dataset:
            img_idxs.append(img_idx)

        img_idxs = list(set(img_idxs))
        PesoTrain.WSI_Data['img_id'] = img_idxs
        PesoTrain.WSI_Data['patch_num'] = np.zeros(len(img_idxs))
        PesoTrain.WSI_Data['grades0'] = np.zeros(len(img_idxs))
        PesoTrain.WSI_Data['grades1'] = np.zeros(len(img_idxs))
        PesoTrain.WSI_Data['grades2'] = np.zeros(len(img_idxs))
        PesoTrain.WSI_Data['o2us'] = np.zeros(len(img_idxs))
        PesoTrain.WSI_Data['fines'] = np.zeros(len(img_idxs))
        

        for i,(x, label, pid, img_idx, global_idx)  in enumerate(PesoTrain.bk_dataset):
            PesoTrain.bk_data['patch_id'].append(int(pid))
            PesoTrain.bk_data['img_id'].append(int(img_idx))
            PesoTrain.bk_data['class'].append(label)
            PesoTrain.bk_data['file_name'].append(bkdf[i][2])
            PesoTrain.bk_data['kmeans_label'].append(0) #init 0


        for x, label, pid, img_idx, global_idx in PesoTrain.train_dataset:
            PesoTrain.WSI_Data['patch_num'][img_idx] += 1
            PesoTrain.sample_data['patch_id'].append(int(pid))
            PesoTrain.sample_data['img_id'].append(img_idx)
            PesoTrain.sample_data['class'].append(label)
            PesoTrain.sample_data['file_name'].append(PesoTrain.files[global_idx][2])
            PesoTrain.sample_data['CAM_file_name'].append(str(global_idx)+".png")
            PesoTrain.sample_data['is_labeled'].append(0)
            # epoch data score
            PesoTrain.sample_data['grades_num'].append(np.zeros(3))
            PesoTrain.sample_data['o2us_num'].append([])
            PesoTrain.sample_data['fines_num'].append([])

        for _, label, pid, imgid, global_idx in PesoTrain.labeled_set:
            PesoTrain.sample_data['is_labeled'][global_idx] = 1
        # 
        PesoTrain.sample_data['scatter_x']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['scatter_y']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['grade']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['o2u']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['heat_score']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['fine']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['noise']=np.zeros(len(PesoTrain.train_dataset))
        PesoTrain.sample_data['kmeans_label']=np.zeros(len(PesoTrain.train_dataset))


        self.logger.info("We have load train dataset:%d, back dataset:%d, labeled dataset:%d, unlabeled dataset:%d, test dataset:%d, files length:%d "
        %(len(PesoTrain.train_dataset),len(PesoTrain.bk_dataset),len(PesoTrain.labeled_set),len(PesoTrain.unlabeled_set),len(PesoTrain.test_dataset),len(PesoTrain.files)))
       
        return PesoTrain.train_dataset, PesoTrain.labeled_set, PesoTrain.unlabeled_set, PesoTrain.test_dataset

    def reset(self, noise_index, add_index_confident):
        self.logger.info(
            "We get %d noise labeled samples and %d informative unlabeled samples"%(len(noise_index),len(add_index_confident)))
        self.logger.info("Before reset,we have %d labeled samples and %d unlabeled samples" % (
            len(PesoTrain.labeled_set), len(PesoTrain.unlabeled_set)))
        
        # noise_index = noise_index.tolist()
        new_labeled_set = []
        new_unlabeled_set = []
        for i, (x, label, patch_idx, img_idx, global_idx) in enumerate(PesoTrain.labeled_set):
            if global_idx in noise_index:
                new_labeled_set.append(
                    (x, PesoTrain.true_labels[global_idx], patch_idx, img_idx, global_idx))
                continue
            else:
                new_labeled_set.append(
                    (x, label, patch_idx, img_idx, global_idx))

        for i, (x, label, patch_idx, img_idx, global_idx) in enumerate(PesoTrain.unlabeled_set):
            if global_idx in add_index_confident:
                new_labeled_set.append(
                    (x, label, patch_idx, img_idx, global_idx))
                if global_idx not in PesoTrain.labeled_set:
                    PesoTrain.labeled_set.append(
                        (x, label, patch_idx, img_idx, global_idx))
            else:
                new_unlabeled_set.append(
                    (x, label, patch_idx, img_idx, global_idx))
        PesoTrain.labeled_set = new_labeled_set
        PesoTrain.unlabeled_set = new_unlabeled_set
        
        # caculate accuracy of noise fliter
        acc_noise=0
        for n in noise_index:
            if n in PesoTrain.noise_idx:
                acc_noise+=1
        self.logger.info("The accuracy of noise sample flitering is [%.3f] "%(acc_noise/len(noise_index))) 
        # caculate number of true labels 
        true_number=0
        for i, (x, label, patch_idx, img_idx, global_idx) in enumerate(PesoTrain.labeled_set):
            if self.true_labels[global_idx]==label:
                true_number+=1

        self.logger.info("After reset, we got true [ %d / %d ] label samples "% (true_number, len(PesoTrain.labeled_set)))
        self.logger.info("After reset, we have %d labeled samples and %d unlabeled samples" % (
            len(PesoTrain.labeled_set), len(PesoTrain.unlabeled_set)))

        return

MASKS = 'D:/DATASETS/hubmap/1000_pixel_images/masks/'
SAVE_PATH='D:/DATASETS/hubmap/1000_pixel_images/result/save_data'

class HubmapTrain(CreateDataInput):
    labeled_loader = None
    unlabeled_loader = None
    valid_loader = None
    train_loader = None
    
    train_dataset= None
    labeled_dataset= None
    unlabeled_dataset= None
    valid_dataset= None

    
    WSI_Data = {
        'img_id': [],
        'patch_num': [],
        'is_labeled':[],
        'noise':[],
        'grade': [],
        'o2us': [],
        'fines': []
    }
    epoch_Data = {
        'epoch': [],
        'iou': [],
        'dice': [],
        'labeled': [],
        'unlabeled': [],
        'noise_in_labeled': [],
        'infor_in_unlabled': []
    }
    sample_data = {
        'img_id': [],
        'patch_id': [],
        'scatter_x': [],
        'scatter_y': [],
        'grade': [],
        'o2u': [],
        'fine': [],
        'swin_score': [],
        'grades_num': [],
        'o2us_num': [],
        'fines_num': [],
        'kmeans_label':[]
    }
    
    def __init__(self,log,config):
        super(HubmapTrain, self).__init__()
        train_df,labeled_df,unlabeled_df, valid_df,noise_masks = make_fold1000(config['organ'],noise_ratio=config['noise_ratio'],seed=config['seed'])
        self.train_df = train_df
        self.labeled_df = labeled_df
        self.unlabeled_df = unlabeled_df
        self.valid_df = valid_df
        
        self.noise_masks = noise_masks
        self.noise_index=[x[0] for x in noise_masks]
        self.organs=config['organ']
        self.log=log
        self.config=config
        log.info("Init dataset numbers: \n\ttrain: %d\n\tlabeled: %d \n\tunlabeled: %d \n\tvalid: %d \n\tnoise: %d"%(
            len(self.train_df),len(self.labeled_df),len(self.unlabeled_df),len(self.valid_df),len(self.noise_masks)
        ))
        
    def init_data(self,shuffle=False):                  
        HubmapTrain.train_dataset=HubmapDataset1000(self.config,self.train_df,self.organs)
        HubmapTrain.labeled_dataset=HubmapDataset1000(self.config,self.labeled_df,self.organs,noise_masks=self.noise_masks)  # insert noise in labeled samples
        HubmapTrain.unlabeled_dataset=HubmapDataset1000(self.config,self.unlabeled_df,self.organs)
        HubmapTrain.valid_dataset = HubmapDataset1000(self.config,self.valid_df, self.organs)
        
        HubmapTrain.train_loader  = DataLoader(
            HubmapTrain.train_dataset,
            batch_size  = self.config['batch_size'],
            sampler = RandomSampler(HubmapTrain.train_dataset) if shuffle else None,
            drop_last   = False,
            shuffle=False,
            num_workers = 8,
            pin_memory  = False,
            worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn = null_collate,
        )
        HubmapTrain.labeled_loader  = DataLoader(
            HubmapTrain.labeled_dataset,
            batch_size  =  self.config['batch_size'],
            sampler = RandomSampler(HubmapTrain.labeled_dataset) if shuffle else None,
            drop_last   = False,
            shuffle=False,
            num_workers = 8,
            pin_memory  = False,
            worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn = null_collate,
        )
        HubmapTrain.unlabeled_loader  = DataLoader(
            HubmapTrain.unlabeled_dataset,
            sampler = RandomSampler(HubmapTrain.unlabeled_dataset) if shuffle else None,
            batch_size  =  self.config['batch_size'],
            drop_last   = False,
            shuffle=False,
            num_workers = 8,
            pin_memory  = False,
            worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn = null_collate,
        )

        HubmapTrain.valid_loader = DataLoader(
            HubmapTrain.valid_dataset,
            batch_size  =  self.config['batch_size'],
            drop_last   = False,
            num_workers = 4,
            pin_memory  = False,
            collate_fn = null_collate,
        )
        
        return 
    def init_save_data(self):
        tile_num=self.config['tile_num']
        #  WSI_Data
        lb_ids=extract_hubmap_data(HubmapTrain.labeled_dataset)
        
        HubmapTrain.WSI_Data['img_id']=[batch['id'] for batch in HubmapTrain.train_dataset]
        HubmapTrain.WSI_Data['is_labeled']=[1 if x in lb_ids else 0 for x in HubmapTrain.WSI_Data['img_id']]
        HubmapTrain.WSI_Data['noise']=[1 if x in self.noise_index else 0 for x in HubmapTrain.WSI_Data['img_id']]
        HubmapTrain.WSI_Data['patch_num']=[tile_num**2]*len(HubmapTrain.train_dataset)
        HubmapTrain.WSI_Data['grade']=np.zeros(len(HubmapTrain.train_dataset))
        HubmapTrain.WSI_Data['o2us']=np.zeros(len(HubmapTrain.train_dataset))
        HubmapTrain.WSI_Data['fines']=np.zeros(len(HubmapTrain.train_dataset))
        
        # sample_data
        for batch in HubmapTrain.train_dataset:
            for i in range(tile_num):
                for j in range(tile_num):
                    HubmapTrain.sample_data['img_id'].append(batch['id'])
                    HubmapTrain.sample_data['patch_id'].append(i*tile_num+j)
                    # epoch data score
                    HubmapTrain.sample_data['grades_num'].append(np.zeros(3))
                    HubmapTrain.sample_data['o2us_num'].append([])
                    HubmapTrain.sample_data['fines_num'].append([])
        lens=len(HubmapTrain.train_dataset)*tile_num**2
        HubmapTrain.sample_data['scatter_x']=np.zeros(lens)
        HubmapTrain.sample_data['scatter_y']=np.zeros(lens)
        HubmapTrain.sample_data['grade']=np.zeros(lens)
        HubmapTrain.sample_data['o2u']=np.zeros(lens)
        HubmapTrain.sample_data['fine']=np.zeros(lens)
        HubmapTrain.sample_data['swin_score']=np.zeros(lens)
        HubmapTrain.sample_data['kmeans_label']=np.zeros(lens)
        
        
    def save_iteration(self,
                       iteration,
                       noise_index,
                       infor_index,
                       CC_label,
                       fine_score_patch,
                       o2u_score_patch,
                       swin_score,
                       kmeans_label,
                       umap_label,
                       best_iou,
                       best_dice,
                       ):
        tile_num=self.config['tile_num']
        
        # update epoch data
        HubmapTrain.epoch_Data['epoch'].append(iteration)
        HubmapTrain.epoch_Data['iou'].append(best_iou)
        HubmapTrain.epoch_Data['dice'].append(best_dice)
        HubmapTrain.epoch_Data['labeled'].append(len(HubmapTrain.labeled_dataset))
        HubmapTrain.epoch_Data['unlabeled'].append(len(HubmapTrain.unlabeled_dataset))
        HubmapTrain.epoch_Data['noise_in_labeled'].append(len(noise_index))
        HubmapTrain.epoch_Data['infor_in_unlabled'].append(len(infor_index))
        
        # update WSIs data
        for k,v in CC_label.items():
            idx=HubmapTrain.WSI_Data['img_id'].index(k)
            HubmapTrain.WSI_Data['grade'][idx]=v
        for k,v in o2u_score_patch.items():
            idx=HubmapTrain.WSI_Data['img_id'].index(k)
            HubmapTrain.WSI_Data['o2us'][idx]=float(np.mean(v))
        for k,v in fine_score_patch.items():
            idx=HubmapTrain.WSI_Data['img_id'].index(k)
            HubmapTrain.WSI_Data['fines'][idx]=float(np.mean(v))
            
        # update sample data
        for k,v in CC_label.items():
            ids=[i for i,val in enumerate(HubmapTrain.sample_data['img_id']) if val==k]
            HubmapTrain.sample_data['grade'][ids]=int(v)
            for i in ids:
                HubmapTrain.sample_data['grades_num'][i][int(v)]+=1
            
        for k,v in o2u_score_patch.items():
            start_idx=HubmapTrain.sample_data['img_id'].index(k)
            for i in range(tile_num):
                for j in range(tile_num):
                    HubmapTrain.sample_data['o2u'][start_idx+i*tile_num+j]=float(v[i*tile_num+j])
                    HubmapTrain.sample_data['o2us_num'][start_idx+i*tile_num+j].append(float(v[i*tile_num+j]))
        for k,v in fine_score_patch.items():
            start_idx=HubmapTrain.sample_data['img_id'].index(k)
            for i in range(tile_num):
                for j in range(tile_num):
                    HubmapTrain.sample_data['fine'][start_idx+i*tile_num+j]=float(v[i*tile_num+j])
                    HubmapTrain.sample_data['fines_num'][start_idx+i*tile_num+j].append(float(v[i*tile_num+j]))
        for k,v in swin_score.items():
            start_idx=HubmapTrain.sample_data['img_id'].index(k)
            for i in range(tile_num):
                for j in range(tile_num):
                    HubmapTrain.sample_data['swin_score'][start_idx+i*tile_num+j]=float(np.max(v[i,j]))
        HubmapTrain.sample_data['swin_score']=normalize(HubmapTrain.sample_data['swin_score'])
        for k,v in kmeans_label.items():
            start_idx=HubmapTrain.sample_data['img_id'].index(k)
            for i in range(tile_num):
                for j in range(tile_num):
                    HubmapTrain.sample_data['kmeans_label'][start_idx+i*tile_num+j]=int(v[i*tile_num+j])
        for k,v in umap_label.items():
            start_idx=HubmapTrain.sample_data['img_id'].index(k)
            for i in range(tile_num):
                for j in range(tile_num):
                    HubmapTrain.sample_data['scatter_x'][start_idx+i*tile_num+j]=float(v[i*tile_num+j][0])
                    HubmapTrain.sample_data['scatter_y'][start_idx+i*tile_num+j]=float(v[i*tile_num+j][1])
                    
        
        if (iteration+1)%4==0 or iteration==0:
            sp=make_dir(os.path.join(self.config['SAVE_PATH'],str(iteration)))
            # save to csv
            pd.DataFrame(HubmapTrain.sample_data).to_csv(
                os.path.join(sp, "sample_data.csv"))
            pd.DataFrame(HubmapTrain.epoch_Data).to_csv(
                os.path.join(sp, "epoch_Data.csv"))
            pd.DataFrame(HubmapTrain.WSI_Data).to_csv(
                os.path.join(sp, "WSI_Data.csv"))
        
    def reset(self,noise_index,infor_index):
        self.log.info("We get %d noise labeled samples and %d informative unlabeled samples"%(len(noise_index),len(infor_index)))
        self.log.info("Before reset,we have \n\tlabeled samples:[%d] \n\tunlabeled samples:[%d] \n\tnoise samples:[%d]" % (len(HubmapTrain.labeled_dataset), len(HubmapTrain.unlabeled_dataset),len(self.noise_masks)))
        
        # update noise
        noise_idxs=[i[0] for i in  self.noise_masks]
        r_noise_idxs=list(set(noise_idxs).intersection(set(noise_index)))
        self.log.info("We fliter [%d] right noise"%len(r_noise_idxs))
        self.noise_masks=list(filter(lambda x : x[0] not in r_noise_idxs, self.noise_masks))
        # add infor 
        for inf in infor_index:
            d=self.unlabeled_df[self.unlabeled_df['id']==int(inf)]
            self.labeled_df=pd.concat((self.labeled_df,d))
            ul_idx=self.unlabeled_df.index.to_list()
            for i,idd in enumerate(self.unlabeled_df['id']):
                if idd == int(inf):
                    self.unlabeled_df.drop(ul_idx[i],axis=0,inplace=True)
        self.init_data()
        self.log.info("After reset,we have \n\tlabeled samples:[%d] \n\tunlabeled samples:[%d] \n\tnoise samples:[%d]" % (len(HubmapTrain.labeled_dataset), len(HubmapTrain.unlabeled_dataset),len(self.noise_masks)))
        
        
        
        
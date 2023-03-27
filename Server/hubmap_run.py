
import torch
import os
import numpy as np
import umap
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from torch.utils.data import SequentialSampler
from torch.nn.parallel import DataParallel
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from Curriculum import CurriculumClustering
from Fine import fine
from O2U import *
from datasets import *
from models.segNet import Net
from utils.log_config import MyLogger
from utils.make_dir import make_dir
from utils.additional import *
from utils.segValidation import *
from utils.serialization import *
from utils.model_utils import *

import warnings
warnings.filterwarnings('ignore')



best_iou=0
best_dice=0
best_swin=None

def init_logger():
    name = "hubmap_largeintestine"
    # name = "hubmap_clean_largeintestine"
    logger = MyLogger(file_name=name)
    logger.info("="*50)
    logger.info("Init Model Config...")
    logger.info("="*50)

    return logger

def init_config(log):
    config={
        'fold':0,
        'cuda_id':'0,1',
        'batch_size':4,
        'seed':2023,
        'tile_num':32,
        'img_size':1024,
        'is_amp':True,
        'noise_ratio':0.2,  # check
        'method':'both',
        'organ':'largeintestine', # [ 'kidney','prostate','largeintestine' ,'spleen','lung']
        
        'pre_iterarion':40,
        'backbone_iterarion':30,
        'num_iteration': 20, #AL
        'o2u_iteration':20,
        'o2u_noise_ratio':0.08,
        'o2u_infor_ratio':0.08,
        
        'Kmeans_Visual_cluster':4,
        'Kmeans_cluster':1,
        'Pseudo_cluster':6,
        'CC_cluster':3,
        
        'TRAIN' :'D:/DATASETS/hubmap/1000_pixel_images/train/',
        'MASKS':'D:/DATASETS/hubmap/1000_pixel_images/masks/',
        'LABELS' :'D:/DATASETS/hubmap/train.csv',
        'OUT_DIR':'D:/DATASETS/hubmap/1000_pixel_images/result',
        'SAVE_PATH':'D:/DATASETS/hubmap/1000_pixel_images/result/save_data'
    }
    for k,v  in config.items():
        log.info("%s : %s"%(k,v))
    return config


def pretrain(config,log,network,dataset,scaler):
    log.info("======= pre-train model ======")
    biou=0
    dataset.init_data(shuffle=True)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=5e-5)
    for epoch in range(config['pre_iterarion']):
        sum_train_loss = 0
        swin={}
        for t, batch in enumerate(dataset.train_loader):
            #  
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].half().cuda()
            batch['mask' ] = batch['mask' ].half().cuda()
            batch['organ'] = batch['organ'].cuda()

            # train
            network.train()
            network.output_type = ['loss']
            with amp.autocast(enabled = is_amp):
                output,encoder,_ = network(batch)
                loss0  = output['bce_loss'].mean()
                loss1  = output['aux2_loss'].mean()
                loss_concat=loss0+0.2*loss1
            optimizer.zero_grad()
            scaler.scale(loss_concat).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            # log infor 
            sum_train_loss += loss_concat.item()
            en=encoder[-1].permute(0,2,3,1)
            for i,idx in enumerate(batch['id']):swin[str(idx)]= en[i]
        # val
        valid_loss = validate1000(network, dataset.valid_loader,config,0)
        if valid_loss[2]>biou:
            biou=valid_loss[2]
            torch.save({
                'state_dict': network.state_dict(),
            }, config['OUT_DIR'] + '/checkpoint/pre.model.pth')
            log.info('save checkpoint ===> pre.model.pth and best iou:[%4.3f]'%(valid_loss[2]))
        # 
        log.info('pre-Iteration:train epoch [%d/%d]:train loss:[%4.3f]; valid loss:[%4.3f]; valid dice:[%4.3f]; valid iou: [%4.3f]; '%(
            epoch,config['pre_iterarion']-1,sum_train_loss/len(dataset.train_dataset),valid_loss[0],valid_loss[1],valid_loss[2], 
        ))


def train_backbone(config,log,network,dataset,iteration,scaler):
    global best_iou
    global best_dice
    global best_swin
    config['batch_size']=4
    dataset.init_data()
    if iteration!=0:
        checkpoint = load_checkpoint(config['OUT_DIR'] + '/checkpoint/%d.model.pth' %  (iteration-1),log)
    else:
        checkpoint = load_checkpoint(config['OUT_DIR'] + '/checkpoint/pre.model.pth',log)
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    torch.save({
            'state_dict': network.state_dict(),
            'iteration': iteration,
        }, config['OUT_DIR'] + '/checkpoint/%d.model.pth' %  (iteration))
    log.info('<=== pre-save checkpoint ===> ')
    # save pre checkpoint
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=5e-5)
    for epoch in range(config['backbone_iterarion']):
        sum_train_loss = 0
        swin={}
        for t, batch in enumerate(dataset.labeled_loader):
            #  
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].half().cuda()
            batch['mask' ] = batch['mask' ].half().cuda()
            batch['organ'] = batch['organ'].cuda()

            # train
            network.train()
            network.output_type = ['loss']
            with amp.autocast(enabled = is_amp):
                output,encoder,_ = network(batch)
                loss0  = output['bce_loss'].mean()
                loss1  = output['aux2_loss'].mean()
                loss_concat=loss0+0.2*loss1
            optimizer.zero_grad()
            scaler.scale(loss_concat).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            # log infor 
            sum_train_loss += loss_concat.item()
            en=encoder[-1].permute(0,2,3,1)
            for i,idx in enumerate(batch['id']):swin[str(idx)]= en[i].cpu().detach().numpy()
        # val
        valid_loss = validate1000(network, dataset.valid_loader,config,iteration)
        if valid_loss[2]>=best_iou:
            best_iou=valid_loss[2]
            best_dice=valid_loss[1]
            best_swin=swin
            torch.save({
                'state_dict': network.state_dict(),
                'iteration': iteration,
            }, config['OUT_DIR'] + '/checkpoint/%d.model.pth' %  (iteration))
            log.info('save checkpoint ===> %d.model.pth and best iou:[%4.3f]'%(iteration,best_iou))
        # 
        log.info('Iteration:[%d], train epoch [%d/%d]:train loss:[%4.3f]; valid loss:[%4.3f]; valid dice:[%4.3f]; valid iou: [%4.3f]; '%(
            iteration,epoch,config['backbone_iterarion']-1,sum_train_loss/len(dataset.labeled_dataset),valid_loss[0],valid_loss[1],valid_loss[2], 
        ))
    
    return

def active_learning_train_epoch(config,logger, network, scaler, iteration, dataset):
    global best_iou
    global best_dice
    global best_swin
    
    logger.info("="*20+"active learning for iteration:%d" % (iteration)+"="*20)
    # init param
    checkpoint = load_checkpoint(config['OUT_DIR'] + '/checkpoint/%d.model.pth' %  (iteration),logger)
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    # extract feature
    fea_cf,features= extract_segfeatures(network,dataset.train_loader)
    train_fea_cf= extract_segfeatures_patch(network,dataset.train_loader,tile_num=config['tile_num'])
    train_ids=extract_hubmap_data(dataset.train_dataset)
    # PCA
    logger.info("PCA & Umap algorithm is running...")
    cf = PCA(n_components=2,copy=False, random_state=config['seed']).fit_transform(fea_cf)
    train_cf = PCA(n_components=2,copy=False, random_state=config['seed']).fit_transform(train_fea_cf)
    # umap and kmeans for visual
    up = umap.UMAP(n_components=2,min_dist=0.005, n_neighbors=20)
    v_cf = up.fit_transform(train_cf)
    v_km = KMeans(n_clusters=config['Kmeans_Visual_cluster'],
                random_state=config['seed']).fit(v_cf)
    v_km_label = v_km.labels_    
    ### CC
    logger.info("Kmeans algorithm is running...")
    km = KMeans(n_clusters=config['Kmeans_cluster'],
                random_state=config['seed']).fit(cf)
    target_label = km.labels_
    logger.info("Cluster result:"+str(numCount(target_label)))
    logger.info("Curriculum algorithm is running...")
    CC = CurriculumClustering(
        n_subsets=config['CC_cluster'], verbose=True, random_state=config['seed'])
    CC.fit(cf, target_label)
    grade_label = CC.output_labels
    CC_label={}
    umap_label={}
    kmeans_label={}
    for i,l in enumerate(train_ids): 
        CC_label[l]=grade_label[i]
        umap_label[l]=v_cf[i*config['tile_num']**2:(i+1)*config['tile_num']**2]
        kmeans_label[l]=v_km_label[i*config['tile_num']**2:(i+1)*config['tile_num']**2]
    ### Fine
    logger.info("Fine Sample algorithm is running...")
    label_fea_cf= extract_segfeatures_patch(network,dataset.labeled_loader,tile_num=config['tile_num'])
    # make Pseudo labels
    pseudo_cf=PCA(n_components=2,copy=False, random_state=config['seed']).fit_transform(label_fea_cf)
    pseudo_labels=KMeans(n_clusters=config['Pseudo_cluster'],random_state=config['seed']).fit(pseudo_cf).labels_
    clean_lab, fine_score = fine(label_fea_cf, pseudo_labels)
    clean_lab = clean_lab.tolist()
    cl_,fi_s=patch2image_fine(clean_lab,fine_score)
    fine_noise_index = find_fine_noise_hubmap(
        clean_label=cl_, fine_score=fi_s, labeled_set=dataset.labeled_dataset)
    labeled_ids=extract_hubmap_data(dataset.labeled_dataset)
    labeled_fine={}
    for l in labeled_ids: labeled_fine[l]=[]
    for i in range(len(fine_score)//config['tile_num']**2): 
        labeled_fine[labeled_ids[i]]=fine_score[i*config['tile_num']**2:(i+1)*config['tile_num']**2]
    
    ### O2U
    config['batch_size']=2
    dataset.init_data()
    logger.info("O2U algorithm is running...")
    o2u=O2Uhub(
        config=config,
        model=network,
        scaler=scaler,
        logger=logger,
        dataset=dataset,
        all_dataset_len=len(dataset.train_dataset),
        grade_label=CC_label
        )
    o2u_noise_index, add_index_confident, o2u_score_patch,o2u_score=o2u.train()
    
    if config['method'] == 'fine':
        all_inters_index = fine_noise_index
        all_inters_index = all_inters_index[:0.2 *int(len(all_inters_index))]
    elif config['method'] == 'o2u':
        all_inters_index = o2u_noise_index
    elif config['method'] == 'both':
        fine_noise_index = fine_noise_index[:int(
            0.1*len(fine_noise_index))]
        all_inters_index = list(
            set(fine_noise_index).union(set(o2u_noise_index)))

    # normalize
    labeled_fine=normalize_patch_dict(labeled_fine)
    o2u_score_patch=normalize_patch_dict(o2u_score_patch)
    best_swin=normalize_patch_dict(best_swin)

    ### save
    dataset.save_iteration(
        iteration,
        all_inters_index,
        add_index_confident,
        CC_label,
        labeled_fine,
        o2u_score_patch,
        best_swin,
        kmeans_label,
        umap_label,
        best_iou,
        best_dice)
    ### reset
    dataset.reset(all_inters_index,add_index_confident)
    

if __name__ == '__main__':

    ##
    log=init_logger()
    config=init_config(log)
    for f in ['checkpoint','train','valid','save_data'] : os.makedirs(config['OUT_DIR'] +'/'+f, exist_ok=True)
    device_ids = [0,1]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    ## dataset ----------------------------------------
    log.info('** dataset setting **')
    hub_data=HubmapTrain(log=log,config=config)
    hub_data.init_data()
    hub_data.init_save_data()
    ## net ----------------------------------------
    log.info('** net setting **')
    scaler = amp.GradScaler(enabled = config['is_amp'])
    net = Net().cuda(device=device_ids[0])
    net = DataParallel(net,device_ids=device_ids)
    ## train ----------------------------------
    log.info('** start training here! **')
    # pre-train
    pretrain(config,log,net,hub_data,scaler)
    for iteration in range(config['num_iteration']):
        train_backbone(config,log,net,hub_data,iteration,scaler)
        active_learning_train_epoch(config,log,net,scaler,iteration,hub_data)

    # end
    torch.cuda.empty_cache()
    

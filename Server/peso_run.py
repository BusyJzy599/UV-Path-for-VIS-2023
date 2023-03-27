from utils.log_config import MyLogger
import os
import torch
from torch.autograd import Variable
from utils.model_utils import *
from utils.serialization import *
from utils.make_dir import make_dir
import numpy as np
import models
import os.path as osp
from torch.cuda.amp import autocast, GradScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from datasets import HDF5Dataset
import pandas as pd
import yaml
from datasets import *
from O2U import O2U
from Curriculum import CurriculumClustering
from Fine import fine
from sklearn.preprocessing import MinMaxScaler
from CAM import *
import cv2
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import pynvml


#
# CONFIG_PATH = "/home/chenzhongming/TVCG_wsi/config/config.yaml"
CONFIG_PATH = "D:\Code\TVCG\project\FlaskServer\config\config.yaml"


def read_config():
    fs = open(CONFIG_PATH, encoding="UTF-8")
    datas = yaml.load(fs)
    return datas


def init_logger(config):
    name = config['dataset']['name']+"_batch_"+str(config["batch_size"])+"_tileSize_"+str(
        config['dataset']["tile_size"])+"_noise_"+str(config["noise_rate"])
    logger = MyLogger(file_name=name, config=config)
    logger.info("="*50)
    logger.info("Init Model Config...")
    logger.info("="*50)
    for k, v in config.items():
        logger.info("%s:%s" % (k, v))
    logger.info("="*50)

    return logger

def init_backbone(config, logger):
    model = models.create(
        config['model'], num_features=config["num_features"], num_classes=config["dataset"]["num_class"])
    if config['multi_cuda']:
        # model=torch.nn.DataParallel(model)
        logger.warning("multi cuda test")
    model = model.cuda()
    scaler = GradScaler()
    if torch.cuda.is_available():
        logger.info("cuda is available")
    else:
        logger.warning(" is not available!")

    return model, scaler


def train_backbone(network, scaler, dataset, epoch, config, logger):
    if config['optim'] == "SGD":
        optimizer = torch.optim.SGD(
            network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(
            network.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(
        reduction='none', ignore_index=-1).cuda()
    #
    labeled_loader = torch.utils.data.DataLoader(dataset=Preprocessor(dataset.labeled_set),
                                                 batch_size=config["batch_size"],
                                                 num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=Preprocessor(dataset.test_dataset),
                                              batch_size=1,
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    bk_loader= torch.utils.data.DataLoader(dataset=Preprocessor(dataset.bk_dataset),
                                              batch_size=1,
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    # next train
    best_auc = 0
    best_acc = 0
    best_cm ,prediction= None,None
    # dynamic train iteration
    # dynamic_epoch=config["pretrained_iteration"]+(epoch//7)*5
    dynamic_epoch = config["pretrained_iteration"]
    for e in range(dynamic_epoch):
        # train models
        network.train()
        for i, (images, labels, _, _, global_idx) in enumerate(labeled_loader):
            images = Variable(images).cuda()
            labels = labels.long().cuda()
            ##
            optimizer.zero_grad()
            #
            with autocast():
                _, logits = network(images)
                loss_1 = criterion(logits, labels)
                loss_1 = loss_1.mean()
                # optimizer.zero_grad()
                # loss_1.backward()
                # optimizer.step()
            ##
            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            #
        with torch.no_grad():
            auc_ = evaluate_auc(test_loader, network)
            acc_, cm = evaluate_acc(test_loader, network)
        logger.info("[Backbone Training] Resnet train in epoch: %d/%d, train_loss: %f, test_auc: %f, test_acc: %f" % (
            e+1, dynamic_epoch, loss_1.item(), auc_, acc_
        ))
        if auc_ > best_auc:
            # save train
            save_checkpoint({
                'state_dict': network.state_dict(),
                'best_auc': auc_,
            }, True,
                fpath=osp.join(config["save_param_dir"], 'pretrained_resnet.pth.tar' if epoch == 0 else (
                    "epoch_"+str(epoch-1)+'_model_best.pth.tar')),
                logger=logger)
            # predict back
            prediction=predict_bk(bk_loader, network)
            # save best
            best_auc = auc_
            best_acc = acc_
            best_cm = cm

    return best_auc, best_acc, best_cm,prediction


def active_learning_train_epoch(config, logger, network, scaler, epoch, input, auc, acc, cm,pre):
    logger.info("="*20+"active learning for epoch:%d" % (epoch)+"="*20)
    # # 初始化所有数据
    train_loader = torch.utils.data.DataLoader(dataset=Preprocessor(input.train_dataset),
                                               batch_size=config["o2u_batch_size"],
                                               num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    labeled_loader = torch.utils.data.DataLoader(dataset=Preprocessor(input.labeled_set),
                                                 batch_size=config["o2u_batch_size"],
                                                 num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    unlabeled_loader = torch.utils.data.DataLoader(dataset=Preprocessor(input.unlabeled_set),
                                                   batch_size=config["o2u_batch_size"],
                                                   num_workers=config['num_workers'], shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=Preprocessor(input.test_dataset),
                                              batch_size=1,
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    bk_loader= torch.utils.data.DataLoader(dataset=Preprocessor(input.bk_dataset),
                                              batch_size=config["o2u_batch_size"],
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    #
    # init param
    checkpoint = load_checkpoint(
        os.path.join(config["save_param_dir"], 'pretrained_resnet.pth.tar' if epoch == 1 else "epoch_"+str(epoch-1)+"_model_best.pth.tar"), logger)
    network.load_state_dict(checkpoint['state_dict'], strict=False)

    fea, _, _, _ = extract_features(
        model, train_loader, logger=logger)
    bk_fea,_,_,_=extract_features(
        model, bk_loader, logger=logger)
    fea_cf = torch.stack(fea).numpy()
    bk_fea_cf = torch.stack(bk_fea).numpy()
    union_fea=np.vstack((fea_cf,bk_fea_cf))
  
    logger.info("PCA algorithm is running...")
    pca = PCA(n_components=config['PCA_components'],
              copy=False, random_state=config['seed'])
    cf=pca.fit_transform(union_fea)
    
    # 
    if config['visual_method'] == 'tsne':
        tsne = TSNE(n_components=2, init='pca', random_state=config['seed'])
        v_cf= tsne.fit_transform(cf)
    elif config['visual_method'] == 'umap':
        up = umap.UMAP(n_components=2, min_dist=0.02, n_neighbors=60)
        v_cf= up.fit_transform(cf)

    v_km = KMeans(n_clusters=config['Kmeans_Visual_cluster'],
                  random_state=config['seed']).fit(v_cf)
    v_km_label = v_km.labels_[:fea_cf.shape[0]]
    bk_v_km_label = v_km.labels_[fea_cf.shape[0]:]
  
    logger.info("Kmeans algorithm is running...")
    # km = DBSCAN(eps=0.5, min_samples=10).fit(cf) 
    km = KMeans(n_clusters=config['Kmeans_cluster'],
                random_state=config['seed']).fit(cf)
    target_label = km.labels_
    logger.info("cluster result:"+str(numCount(target_label)))

    logger.info("Curriculum algorithm is running...")
    CC = CurriculumClustering(
        n_subsets=config['CC_cluster'], verbose=True, random_state=config['seed'])
    CC.fit(cf, target_label)
    grade_label = CC.output_labels

    logger.info("Fine Sample algorithm is running...")
    labeled_fea, _, _, _labels = extract_features(
        model, labeled_loader, logger=logger)
    labeled_fea = torch.stack(labeled_fea).numpy()
    _labels = torch.stack(_labels).numpy()
    clean_lab, fine_score = fine(labeled_fea, _labels)
    clean_lab = clean_lab.tolist()
    fine_noise_index = find_fine_noise(
        clean_label=clean_lab, fine_score=fine_score, labeled_set=input.labeled_set)
    # normalize
    fine_score = normalize(fine_score)
    #
    fine_score_global = np.zeros(len(input.train_dataset))
    for i, (_, label, pid, imgid, global_idx) in enumerate(input.labeled_set):
        fine_score_global[global_idx] = fine_score[i]

    logger.info("O2U algorithm is running...")
    o2u = O2U(model=network,
              scaler=scaler,
              epoch=epoch,
              config=config,
              logger=logger,
              labeled_data_loader=labeled_loader,
              unlabeled_data_loader=unlabeled_loader,
              test_data_loader=test_loader,
              all_dataset_len=len(input.train_dataset),
              grade_label=grade_label)
    o2u_noise_index, add_index_confident, o2u_score = o2u.train()
    # normalize
    o2u_score = normalize(o2u_score)
    #
    if config["methods"] == 'fine':
        all_inters_index = fine_noise_index
        all_inters_index = all_inters_index[:0.2 *
                                            int(len(all_inters_index))]
    elif config["methods"] == 'o2u':
        all_inters_index = o2u_noise_index
        # all_inters_index = all_inters_index[:int(
        #     0.2*len(all_inters_index))]
    elif config["methods"] == 'both':
        fine_noise_index = fine_noise_index[:int(
            0.1*len(fine_noise_index))]
        # o2u_noise_index = o2u_noise_index[:int(0.1*len(o2u_noise_index))]
        all_inters_index = list(
            set(fine_noise_index).union(set(o2u_noise_index)))

    # knn
    if config['Knn_act'] == 1:
        noise_feaure = []
        logger.info("K neighbors algorithm is running...")
        for i, (x, label, index, _, global_idx) in enumerate(input.labeled_set):
            if global_idx in all_inters_index:
                noise_feaure.append(labeled_fea[i])

        k_neighbors = find_Neighbors(labeled_fea,  noise_feaure, k_num=1)
        k_neighbors.tolist()
        k_neighbors = np.unique(k_neighbors)
        k_neig_index = []
        for i in range(int(len(input.labeled_set))):
            if i in k_neighbors:
                k_neig_index.append(input.labeled_set[i][4])
        all_union_index = list(set(all_inters_index).union(set(k_neig_index)))
    else:
        all_union_index = all_inters_index

    # gradCAM
    if config['grad_save'] == 1 and epoch >= config["max_iteration"]-1:
        logger.info("Grad-CAM images are generated...")
        grad_cam = GradCam(model, target_layer_names=[
                           'base'], img_size=config['dataset']['size'])
        for img, label, pid, imgid, global_idx in input.train_dataset:
            img = np.asarray(img).reshape(
                config['dataset']['size'], config['dataset']['size'], 3)
            img = np.float32(cv2.resize(
                img, (config['dataset']['size'], config['dataset']['size']))) / 255
            pre_img = preprocess_image(img)
            pre_img.required_grad = True
            target_index = None
            mask = grad_cam(pre_img, target_index)
            CAM_save_path = make_dir(os.path.join(config['save_data_dir'], os.path.join(
                'init_data_image', 'image_CAM_'+str(imgid))))
            CAM_save_path = os.path.join(CAM_save_path, str(global_idx)+".png")
            show_cam_on_image(img, mask, CAM_save_path)

    # save
    save_iteration_data(input, epoch, all_union_index, add_index_confident,
                        v_cf, grade_label, o2u_score, fine_score_global, acc, auc, cm, pre,v_km_label,bk_v_km_label)
    # reset operation need to be rectified
    input.reset(all_union_index, add_index_confident)

    return epoch, input.sample_data, input.epoch_Data, input.WSI_Data


def save_iteration_data(
        dataset, epoch, noise_index, infor_index, visual_cf, grade_label, o2u_score, fine_score_global, acc, auc, cm,pre, km,bk_km):
    for i, (_, label, pid, imgid, global_idx) in enumerate(dataset.train_dataset):
        dataset.sample_data['scatter_x'][global_idx] = visual_cf[global_idx][0]
        dataset.sample_data['scatter_y'][global_idx] = visual_cf[global_idx][1]
        dataset.sample_data['grade'][global_idx] = grade_label[global_idx]
        dataset.sample_data['o2u'][global_idx] = o2u_score[global_idx]
        dataset.sample_data['kmeans_label'][global_idx] = km[global_idx]

        dataset.sample_data['grades_num'][global_idx][grade_label[global_idx]] += 1
        dataset.sample_data['o2us_num'][global_idx].append(
            float(o2u_score[global_idx]))
        dataset.sample_data['heat_score'][global_idx] = 0

        if global_idx in noise_index or global_idx in infor_index:
            dataset.sample_data['noise'][global_idx] = 1
        # default unlabeled
        # sample_data['is_labeled'].append(0)
        dataset.sample_data['fine'][global_idx] = fine_score_global[global_idx]
        dataset.sample_data['fines_num'][global_idx].append(
            fine_score_global[global_idx])

        # update WSI data
        dataset.WSI_Data['grades'+str(grade_label[global_idx])][imgid] += 1
        dataset.WSI_Data['o2us'][imgid] += o2u_score[global_idx]
        dataset.WSI_Data['fines'][imgid] += fine_score_global[global_idx]

    # update epoch data
    dataset.epoch_Data['epoch'].append(epoch)
    dataset.epoch_Data['acc'].append(acc)
    dataset.epoch_Data['auc'].append(auc)
    dataset.epoch_Data['labeled'].append(len(dataset.labeled_set))
    dataset.epoch_Data['unlabeled'].append(len(dataset.unlabeled_set))
    dataset.epoch_Data['noise_in_labeled'].append(len(noise_index))
    dataset.epoch_Data['infor_in_unlabled'].append(len(infor_index))

    # update WSIs data
    for imgid in dataset.WSI_Data['img_id']:
        for i in range(3):
            dataset.WSI_Data['grades' +
                             str(i)][imgid] /= dataset.WSI_Data['patch_num'][imgid]
        dataset.WSI_Data['o2us'][imgid] /= dataset.WSI_Data['patch_num'][imgid]
        dataset.WSI_Data['fines'][imgid] /= dataset.WSI_Data['patch_num'][imgid]

    # updata back data
    for i, (_, label, pid, imgid, global_idx) in enumerate(dataset.bk_dataset):
        dataset.bk_data['class'][global_idx] = int(pre[global_idx])
        dataset.bk_data['kmeans_label'][global_idx] = bk_km[global_idx]
        
    name = config['dataset']['name']+"_batch_"+str(config["batch_size"])+"_tileSize_"+str(
        config['dataset']["tile_size"])+"_noise_"+str(config["noise_rate"])
    base_path = os.path.join(config['save_data_dir'], "save_data/"+name)
    base_path = make_dir(base_path)

    # save to csv
    pd.DataFrame(dataset.sample_data).to_csv(
        os.path.join(base_path, "sample_data.csv"))
    pd.DataFrame(dataset.bk_data).to_csv(
        os.path.join(base_path, "bk_data.csv"))
    pd.DataFrame(dataset.epoch_Data).to_csv(
        os.path.join(base_path, "epoch_Data.csv"))
    pd.DataFrame(dataset.WSI_Data).to_csv(
        os.path.join(base_path, "WSI_Data.csv"))
    pd.DataFrame(cm).to_csv(
        os.path.join(base_path, "confusion.csv"))
 

    return dataset.sample_data, dataset.epoch_Data, dataset.WSI_Data


def getCudaMemoryRate(cuda_ids):
    while 1:
        for cu_id in cuda_ids:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(cu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            m_used = meminfo.used//(1024**2)
            if m_used < 2000:
                return str(cu_id)
        time.sleep(1)


if __name__ == '__main__':
   
  # read config
    config = read_config()
    # init logger
    logger = init_logger(config)
    # init dataset
    peso_data = PesoTrain(config, logger)
    peso_data.init_data()

    # ===========================================
    logger.info("Loading cuda....")
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_id']
    logger.info("Start!!!")
    # ===========================================

    # create model
    model, scaler = init_backbone(config, logger)
    # train
    train_backbone(model, scaler, peso_data, 0, config, logger)
    # AL
    # for e in range(1, config['max_iteration']+1):
    #     auc, acc, cm = train_backbone(
    #         model, scaler, peso_data, e, config, logger)
    #     # epoch 1
    #     _, s, e, w = active_learning_train_epoch(
    #         config, logger, model, scaler, e, peso_data, auc, acc, cm)

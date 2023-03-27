from __future__ import absolute_import,print_function
from collections import OrderedDict
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import time
import torch
import numpy as np
import random
from numpy.testing import assert_array_almost_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import cv2
import os
import torch.cuda.amp as amp


MASKS = 'D:/DATASETS/hubmap/1000_pixel_images/masks/'
def noisify_seg(df):
    msks=[]
    for d in range(len(df)):
        m=cv2.imread(os.path.join(MASKS,df.iloc[d].organ,str(df.iloc[d].id)+".png"),cv2.IMREAD_GRAYSCALE).astype(np.float32)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i,j]=1-m[i,j]
        msks.append([
            str(df.iloc[d].id),
            df.iloc[d].organ,
            m
        ])
    return msks

# 0,1分类 相邻两类做noise
def noisify_binary(labels, ratio=0.3):
    import copy
    x = copy.deepcopy(labels.flatten().tolist())
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
    res=np.array(x).reshape(-1,1)
    diff=np.sum(res!=labels)/len(x)
    return np.array(x).reshape(-1,1),diff
# 噪声生成模块
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y
# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise
    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    return y_train, actual_noise
def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n
        
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    
    return y_train, actual_noise
def noisify(dataset='kather', nb_classes=3, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'nearly':
        train_noisy_labels, actual_noise_rate = noisify_binary(train_labels, noise_rate)
    return train_noisy_labels, actual_noise_rate

# 
def normalize(x):
    x=np.array(x).tolist()
    m_sorted = sorted(enumerate(x), key=lambda x:x[1])
    sorted_inds = [m[0] for m in m_sorted]
    tool=MinMaxScaler(feature_range=(0,1))
    sorted_inds=tool.fit_transform(np.array(sorted_inds).reshape(-1,1)).reshape(1,-1).tolist()
    return sorted_inds[0]

def normalize_dict(d):
    minV=np.min(list(d.values()))
    maxV=np.max(list(d.values()))
    basic=maxV-minV
    for k in d.keys():
        d[k]=(d[k]-minV)/basic
    return d
def normalize_patch_dict(d):
    smooth=1e-5
    for k,v in d.items():
        k_max=np.max(v)
        k_min=np.min(v)
        v_=(v-k_min)/(k_max-k_min+smooth)
        d[k]=v_
    return d

    

# 计算acc以及auc
def evaluate_auc(test_loader, model):
    model.eval()
    prediction=[]
    llabel=[]
    for images, labels, _,_ ,_ in test_loader:
        images = Variable(images).cuda()
        #print images.shape
        _, logits1 = model(images)
        outputs1 = F.softmax(logits1, dim=1)
        # pred1, _ = torch.max(outputs1.data.cpu(), 1)
        for pre, label in zip(outputs1.data.cpu(), labels):
            llabel.append(label)
            prediction.append(pre.numpy())
    llabel = np.array(llabel)
    # llabel = label_binarize(llabel, classes=[0, 1])
    prediction = np.array(prediction)
    # AUC = roc_auc_score(llabel, prediction, multi_class='ovo', average='micro')#, multi_class='ovr'
    # fpr, tpr, _ = roc_curve(llabel.ravel(), prediction.ravel())
    fpr, tpr, _ = roc_curve(llabel, prediction[:,1], pos_label=1)
    AUC= auc(fpr, tpr)
    model.train()
    return AUC

def evaluate_acc(test_loader, model):
    model.eval()
    correct1 = 0
    total1 = 0
    true_label=[]
    pre_label=[]
    for images, labels, _ ,_ ,_ in test_loader:
        images = Variable(images).cuda()
        #print images.shape
        _, logits1 = model(images)
        outputs1 = F.log_softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
        # 
        true_label.extend(list(labels))
        pre_label.extend(list(pred1.cpu()))
    
        
    cm = confusion_matrix(np.array(true_label), np.array(pre_label))
    # acc1 = 100 * float(correct1) / float(total1)
    acc1 = float(correct1) / float(total1)
    model.train()
    return acc1,cm

def predict_bk(back_loader,model):
    model.eval()
    prediction=[]
    for images, _, _,_ ,_ in back_loader:
        images = Variable(images).cuda()
        #print images.shape
        _, logits1 = model(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        prediction.extend(list(np.array(pred1.cpu())+1))
    return prediction

def predict_train(loader,model):
    model.eval()
    prediction=np.zeros(20000)
    for images, _, _,_ ,global_idx in loader:
        images = Variable(images).cuda()
        #print images.shape
        _, logits1 = model(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        pred=list(np.array(pred1.cpu())+1)
        for idx,pre in zip(global_idx,pred):
            prediction[idx]=pre
    # print(prediction)
    return prediction



# 特征提取
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def extract_segfeatures(net, data_loader):
    net = net.eval()
    features_v = []
    features=[]
    for t, batch  in enumerate(data_loader):
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled=True):
                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['mask'] = batch['mask'].cuda()
                output,encoder,_ = net(batch)
                encoder=encoder[-1].cpu().numpy()
        for f in encoder:
            features_v.append(f.flatten())
            features.append(f)
    return np.array(features_v),np.array(features)
           
def extract_segfeatures_patch(net, data_loader,tile_num=32):
    net = net.eval()
    features_v = []
    for t, batch  in enumerate(data_loader):
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled=True):
                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['mask'] = batch['mask'].cuda()
                output,encoder,_ = net(batch)
                encoder=encoder[-1].cpu().numpy()
        for f in encoder:
            f=f.transpose(1,2,0)
            for i in range(tile_num):
                for j in range(tile_num):
                    features_v.append(f[i,j])
    return np.array(features_v)
           
def patch2image_fine(clean_lab,fine_score,tile_num=32):
    score=[]
    cl=[]
    for i in range(len(fine_score)//(tile_num**2)):
        img_s=np.array(fine_score[i*tile_num**2:(i+1)*tile_num**2])
        cl_s=np.array(clean_lab[i*tile_num**2:(i+1)*tile_num**2])
        score.append(img_s.mean())
        cl.append(cl_s.mean())
        
    return cl,score
              

def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()

    if modules is None:
        outputs, probs = model(inputs)
        outputs = outputs.data.cpu()
        probs = probs.data.cpu()
        return outputs, probs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())


def extract_features(model, data_loader, logger):
    model.eval()
    features = []
    labels = []
    probss = []
    probsss = []
    start = time.time()
    with torch.no_grad():
        for i, (imgs, lb, pid,imid,global_idx) in enumerate(data_loader):
            outputs, probs = extract_cnn_feature(model, imgs)
            probs = F.softmax(probs, dim=1)
            for output, prob, label_ in zip(outputs, probs, lb):
                features.append(output)
                probss.append(torch.argmax(prob))
                probsss.append(prob[int(label_)])
                labels.append(label_)
            

    logger.info("Extract %d features in time:%d s" % (i+1, time.time()-start))
    # return features, img_idx, patch_idx, labels
    return features, probss, probsss, labels



# 最近邻
def find_Neighbors(label_feature, noise_feature, k_num=1):
    neigh = NearestNeighbors(n_neighbors=k_num)
    neigh.fit(label_feature)
    _, neighbors = neigh.kneighbors(noise_feature)
    neighbors = neighbors.reshape(1, -1)
    return neighbors[0]


def find_fine_noise(clean_label,fine_score,labeled_set):
    fine_noise_index = []
    fine_noise_score = []
    fine_index = []
    index_fine_new = []
    for i in range(int(len(labeled_set))):
            if i not in clean_label:
                fine_index.append(i)
                fine_noise_score.append(fine_score[i])
            index_fine_new.append(labeled_set[i][4])
    fine_score_sort = np.argsort(fine_noise_score)
    for i in range(int(len(fine_score_sort))):
        fine_noise_index.append(
            labeled_set[fine_index[fine_score_sort[i]]][4])
    return fine_noise_index


def find_fine_noise_hubmap(clean_label,fine_score,labeled_set):
    fine_noise_index = []
    fine_noise_score = []
    fine_index = []

    for i in range(int(len(labeled_set))):
            if i not in clean_label:
                fine_index.append(i)
                fine_noise_score.append(fine_score[i])

    fine_score_sort = np.argsort(fine_noise_score)
    for i in range(int(len(fine_score_sort))):
        fine_noise_index.append(
            labeled_set[fine_index[fine_score_sort[i]]]["id"])
    return fine_noise_index

def extract_hubmap_data(dataset,key="id"):
    r=[]
    for batch in dataset:
        r.append(batch[key])
    return r

def numCount(L):
    s = set(L)
    result = dict.fromkeys(s, 0)
    for i in L:
        if i in s:
            result[i] += 1
    return result
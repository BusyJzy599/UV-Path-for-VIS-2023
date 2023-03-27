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
def noisify(dataset='kather', nb_classes=2, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
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
    sorted_inds=tool.fit_transform(np.array(sorted_inds).reshape(-1,1))
    return sorted_inds

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
    for images, labels, _ ,_ ,_ in test_loader:
        images = Variable(images).cuda()
        #print images.shape
        _, logits1 = model(images)
        outputs1 = F.log_softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
    # acc1 = 100 * float(correct1) / float(total1)
    acc1 = float(correct1) / float(total1)
    model.train()
    return acc1

# 特征提取
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

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
    # 
    img_idx=[]
    patch_idx=[]

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
                # 
                # img_idx.append(pid)
                # patch_idx.append(imid)

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

# 
# def find_fine_noise(clean_label,fine_score,labeled_set):
#     fine_noise_index = []
#     fine_noise_score = []
#     fine_index = []
#     index_fine_new = []
#     for i in range(int(len(labeled_set))):
#             if i not in clean_label:
#                 fine_index.append(i)
#                 fine_noise_score.append(fine_score[i])
#             index_fine_new.append(labeled_set[i][2])
#     fine_score_sort = np.argsort(fine_noise_score)
#     for i in range(int(len(fine_score_sort))):
#         fine_noise_index.append(
#             labeled_set[fine_index[fine_score_sort[i]]][2])
#     return fine_noise_index

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


def numCount(L):
    s = set(L)
    result = dict.fromkeys(s, 0)
    for i in L:
        if i in s:
            result[i] += 1
    return result
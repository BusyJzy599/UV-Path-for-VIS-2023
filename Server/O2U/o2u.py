from __future__ import absolute_import
from difflib import context_diff
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from utils.model_utils import evaluate_acc
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class O2U(object):
    def __init__(self,
                 model,
                 scaler,
                 config:None,
                 logger:None,
                 epoch:int =0,
                 labeled_data_loader=None,
                 unlabeled_data_loader=None,
                 test_data_loader=None,
                 grade_label: list = None,
                 all_dataset_len: int = 0,
                 ):
        """
        O2U for noise score

        Parameters
        ----------
        model : torch.nn
        max_epoch: the max  number of the iteration
        labeled_data_loader:labeled dataset
        unlabeled_data_loader:unlabeled dataset
        test_data_loader:test dataset
        grade_label: Curriculum algorithm for noise levels list
        all_dataset_len: the length of data
        K1:O2U noise ratio for labeled
        K2:O2U informative ratio for unlabeled

        """
        super(O2U, self).__init__()
        self.model = model
        self.scaler=scaler
        self.logger=logger
        self.max_epoch = config['o2u_iteration']
        self.epoch=epoch
        self.labeled_data_loader = labeled_data_loader
        self.unlabeled_data_loader = unlabeled_data_loader
        self.test_data_loader = test_data_loader

        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss(
            reduce=False, ignore_index=-1).cuda()
        self.all_dataset_len = all_dataset_len
        self.grade_label = grade_label
        #
        self.moving_loss_dic = np.zeros(all_dataset_len)
        self.moving_entropy_dic = np.zeros(all_dataset_len)
        self.K1 = config['o2u_noise_rate']
        self.K2 = config['o2u_infor_rate']

    def train(self):
        #
        all_o2u_scores=np.zeros(self.all_dataset_len, dtype=float)
        # dynamic train iteration
        # dynamic_epoch=self.max_epoch+(self.epoch//6)*5
        dynamic_epoch=self.max_epoch

        for epoch in tqdm(range(dynamic_epoch)):
            global_loss = 0  # loss for o2u
            self.model.train()
            with torch.no_grad():
                acc_ = evaluate_acc(self.test_data_loader, self.model)
            # init example loss&entroy
            example_loss = np.zeros(self.all_dataset_len, dtype=float)
            example_entropy = np.zeros(self.all_dataset_len, dtype=float)
            # update lr
            lr = self.adjust_learning_rate(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            # caculate labeled sample loss
            for i, (images, labels, indexes,_,global_idx) in enumerate(self.labeled_data_loader):
                images = Variable(images).cuda()
                labels = labels.long().cuda()

                # train
                _, logits = self.model(images)
                # caculate sum loss
                loss_1 = self.criterion(logits, labels)
                ##
                # self.scaler.scale(loss_1).backward()
                # self.scaler.step(optimizer)
                # self.scaler.update()
                #
                for pi, cl in zip(global_idx, loss_1):
                    example_loss[pi] = cl.cpu().data.item()
                    all_o2u_scores[pi]=cl.cpu().data.item()
                # update sum loss
                global_loss += loss_1.sum().cpu().data.item()
                loss_1 = loss_1.mean()
                self.optimizer.zero_grad()
                loss_1.backward()
                self.optimizer.step()
            # caculate unlabeled sample entroy
            for i, (images, _, indexs,_,global_idx) in enumerate(self.unlabeled_data_loader):
                images = Variable(images).cuda()
                _, logits = self.model(images)
                p = -F.softmax(logits, dim=1)
                log_p = F.log_softmax(logits, dim=1)
                entropy = torch.mul(p, log_p).sum(1)
                for pi, en in zip(global_idx, entropy):
                    example_entropy[pi] = en.cpu().data.item()
                    all_o2u_scores[pi]=en.cpu().data.item()
            # mean
            example_entropy = example_entropy - example_entropy.mean()
            example_loss = example_loss - example_loss.mean()
            
            # add each loss and entropy
            self.moving_entropy_dic += example_entropy
            self.moving_loss_dic += example_loss
            
            # filter noise in labeled samples
            ind_1_sorted = np.argsort(-self.moving_loss_dic)
            num_remember = int(self.K1 * self.all_dataset_len)  # 噪声样本的占比
            ind_1_sorted_ = ind_1_sorted[: num_remember]
            noise_ind_1_sorted = [
                x for x in ind_1_sorted_ if self.grade_label[x] == 0]
            
    
            # probability
            index_new = []
            moving_loss_dic_new = np.sort(self.moving_loss_dic)[::-1]
            moving_loss_dic_new = moving_loss_dic_new[:len(self.labeled_data_loader)]
            pro = (moving_loss_dic_new-moving_loss_dic_new.min(0)) / \
                (moving_loss_dic_new.max(0)-moving_loss_dic_new.min(0))
            for i in range(int(len(self.labeled_data_loader))):
                index_new.append(ind_1_sorted[i])
            
            # filter informative in unlabeled samples
            ind_2_sorted = np.argsort(-self.moving_entropy_dic)
            # ind_2_sorted = ind_2_sorted[:len(self.unlabeled_data_loader)]
            ind_2_sorted.tolist().reverse()
            ea_mi_number = int(self.K2 * self.all_dataset_len)  # 简单样本的比例
            ind_2_sorted = ind_2_sorted[:ea_mi_number]
            ea_mi_ind_2_sorted = [x for x in ind_2_sorted if self.grade_label[x] == 0]

        return noise_ind_1_sorted, ea_mi_ind_2_sorted,all_o2u_scores
        # return noise_ind_1_sorted, pro, index_new, ea_mi_ind_2_sorted

            

    @staticmethod
    def adjust_learning_rate(epoch):
        t = ((epoch-1) % 5 + 1) / float(5)
        lr = (1 - t) * 0.01 + t * 0.001
        return lr

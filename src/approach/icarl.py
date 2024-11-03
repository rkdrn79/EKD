import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform

import wandb
import numpy as np
import time


class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, #load_model0=True,
                 logger=None, exemplars_dataset=None, T=2, lamb=1,
                 dkd_control = 'deterministic', dkd_switch = 'all',dkd_shape='one',dkd_m=1.0,distill_percent=0.2,
                 ikr_control = 'none', pk_approach='average',ikr_switch='all',ikr_m=1.0,recycle_percent=0.8
                 ):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train,  logger, exemplars_dataset, #load_model0,
                                   dkd_control,dkd_switch,dkd_shape,dkd_m,distill_percent,
                                   ikr_control,pk_approach,ikr_switch,ikr_m,recycle_percent)
        
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.dkd_switch_array_update=[0]*self.nepochs

        self.dkd_cnt = 0
        self.dkd_time = 0
        self.remaining_time = 0


        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device), return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader, tst_loader):
        """Contains the epochs loop"""

        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader,tst_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # Algorithm 4: iCaRL ConstructExemplarSet and Algorithm 5: iCaRL ReduceExemplarSet
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader, e):
        start_time = time.time()
        self.dkd.current_task = t
        """Runs a single epoch"""
        # erf_kd_use, rgr_kd_use = self.cycle._get_distill_use(e)
        dkd_activation = self.dkd._switch_function(e)
        self.dkd_switch_array_update[e] = dkd_activation

        if self.ikr_control == 'deterministic':
            if self.dkd_control == 'adaptive':
                self.ikr._dkd_adaptive_update(self.dkd.nai_s,self.dkd.ndi_s,self.dkd.Ni_s,self.dkd.li_s,self.dkd.gi_s,self.dkd.thresholds,
                                            self.dkd_switch_array_update)
            ikr_activation = self.ikr._switch_function(e, dkd_activation)
        elif self.ikr_control == 'none':
            ikr_activation = 0

        total_num = 0

        total_total_loss = 0
        total_train_loss = 0
        total_kd_loss = 0
        # total_weight = 0
        total_dkd_weight = 0
        total_ikr_loss = 0

        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        # self.iter_num = len(trn_loader)

        for images, targets in trn_loader:
            # Forward old model
            outputs_old = None

            if t > 0 and dkd_activation == True:
                outputs_old = self.model_old(images.to(self.device))

            # Forward current model
            outputs = self.model(images.to(self.device))

            ## 수정 필요
            total_loss, train_loss, kd_loss, weight = self.criterion(t, outputs, targets.to(self.device), outputs_old, epoch = e, dkd_activation = dkd_activation, ikr_activation = ikr_activation)

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            #
            if t>0 :
                total_loss, train_loss, kd_loss = torch.tensor(total_loss),torch.tensor(train_loss),torch.tensor(kd_loss)
                self.dkd._save_distill_weight(total_loss = total_loss.item(), train_loss = train_loss.item(), kd_loss = kd_loss.item(), epoch = e, t=t)
                if self.ikr_control=='deterministic':
                    self.ikr._save_distill_weight(total_loss = total_loss.item(), train_loss = train_loss.item(), kd_loss = kd_loss.item(), epoch = e, t=t)# ,iter_num = self.iter_num)

            #================= Log =================#
            total_num += 1
            total_total_loss += total_loss.item()
            total_train_loss += train_loss.item()
            # if t > 0 and dkd_activation:
            #     total_kd_loss += kd_loss.item()
            #     total_weight += weight
            if t > 0 and dkd_activation:
                total_kd_loss += kd_loss.item()
                total_dkd_weight += weight
            elif t>0 and ikr_activation:
                total_ikr_loss += weight

        end_time = time.time()
        time_consumed = np.round(end_time-start_time,4)
        if dkd_activation:
            self.dkd_time += time_consumed
            self.dkd_cnt +=1
        else:
            self.remaining_time += time_consumed

        # print("| Epoch: ", e + 1, "Loss: ", total_total_loss / total_num, "Train Loss: ", total_train_loss / total_num, "KD Loss: ", total_kd_loss / total_num, "KD Weight: ", total_weight / total_num)

        print("| Epoch: ", e + 1, 
            "Loss: ", np.round(total_total_loss / total_num,5), 
            "Train Loss: ", np.round(total_train_loss / total_num,5), 
            "KD Loss: ", np.round(total_kd_loss / total_num,5),
            "DKD Weight: ", np.round(total_dkd_weight / total_num,5),
            "IKR Loss: ",np.round(total_ikr_loss / total_num,5), 
        )
        # Log wandb task t
        wandb.log({"epoch": e,
                "task {} Traing total_loss".format(t): np.round(total_total_loss / total_num,5),
                "task {} Traing train_loss".format(t): np.round(total_train_loss / total_num,5), 
                "task {} Traing kd_loss".format(t): np.round(total_kd_loss / total_num,5), 
                "task {} Traing dkd_weight".format(t): np.round(total_dkd_weight / total_num,5),
                "task {} Traing ikr_loss".format(t): np.round(total_ikr_loss / total_num,5)})



    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            # total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            total_loss, train_loss, kd_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs, feats = self.model(images.to(self.device), return_features=True)
                # loss,_,_,_ = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                total_l, train_l, kd_l,_ = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                else:
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                # Log
                total_l, train_l, kd_l = torch.tensor(total_l), torch.tensor(train_l), torch.tensor(kd_l)
                # total_loss += loss.item() * len(targets)
                total_loss += total_l.item() * len(targets)
                train_loss += train_l.item() * len(targets)
                if t > 0:   
                    kd_loss += kd_l.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        # return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
        return total_loss / total_num, train_loss/total_num, kd_loss/total_num , total_acc_taw / total_num, total_acc_tag / total_num
    
    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None, epoch = None, dkd_activation=0, ikr_activation=0):
        """Returns the loss value"""

        total_loss = 0
        train_loss = 0
        kd_loss = 0
        weight = 0

        # Classification loss for new classes
        train_loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)


        # # Distillation loss for old classes
        # if t > 0 and dkd_activation == True:
        #     # The original code does not match with the paper equation, maybe sigmoid could be removed from g
        #     g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
        #     q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
        #     kd_loss = self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
        #                             range(sum(self.model.task_cls[:t])))
        #     weight = self.erf._get_distill_weight(epoch, train_loss.item(), kd_loss.item(),self.distill_percent,self.nepochs, self.cycle_approach)
        #     kd_loss *= weight * self.lamb

        # total_loss = train_loss + kd_loss

        # return total_loss, train_loss, kd_loss, weight


        if t > 0 and dkd_activation: ## DKD Activation
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            kd_loss = self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(sum(self.model.task_cls[:t])))
            weight = float(self.dkd._get_distill_weight(epoch))

            total_loss = train_loss + weight * self.lamb * kd_loss

        
        elif t > 0 and ikr_activation: ## IKR Activation
            weight = float(self.ikr._get_distill_weight(epoch,dkd_activation))
            total_loss = train_loss + weight * self.lamb

        elif t > 0 :
            total_loss = train_loss

        elif (t==0) or (dkd_activation == 0 and ikr_activation == 0):
            total_loss = train_loss

        return total_loss, train_loss, torch.tensor(kd_loss), weight
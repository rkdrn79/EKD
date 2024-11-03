import copy
import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import wandb
import numpy as np
import time

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, #load_model0=True,
                 logger=None, exemplars_dataset=None, T=2, 
                 dkd_control = 'deterministic', dkd_switch = 'all',dkd_shape='one',dkd_m=1.0,distill_percent=0.2,
                 ikr_control = 'none', pk_approach='average',ikr_switch='all',ikr_m=1.0,recycle_percent=0.8,
                 lamb=5., lamb_mr=1., dist=0.5, K=2,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train,  logger, exemplars_dataset, #load_model0,
                                   dkd_control,dkd_switch,dkd_shape,dkd_m,distill_percent,
                                   ikr_control,pk_approach,ikr_switch,ikr_m,recycle_percent)
        
        self.lamb = lamb
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamb
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss

        self.dkd_switch_array_update=[0]*self.nepochs
        self.dkd_cnt = 0
        self.dkd_time = 0
        self.remaining_time = 0

        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamb', default=5., type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamb-mr', default=1., type=float, required=False,
                            help='Trade-off for the MR loss (default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')
        # Flags for ablating the approach
        parser.add_argument('--remove-less-forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove-margin-ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code)
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        self.model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in self.model.heads[:-1]])
                                                   / self.model.heads[-1].out_features)
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader,tst_loader):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader,tst_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

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
            images, targets = images.to(self.device), targets.to(self.device)

            # Forward current model
            outputs, features = self.model(images, return_features=True)

            # Forward previous model
            ref_outputs = None
            ref_features = None
            if t > 0 and dkd_activation:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
            total_loss, train_loss, kd_loss, weight = self.criterion(t, outputs, targets, ref_outputs, features, ref_features, epoch = e, dkd_activation = dkd_activation, ikr_activation = ikr_activation)

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if t>0 :
                total_loss, train_loss, kd_loss = torch.tensor(total_loss),torch.tensor(train_loss),torch.tensor(kd_loss)
                self.dkd._save_distill_weight(total_loss = total_loss.item(), train_loss = train_loss.item(), kd_loss = kd_loss.item(), epoch = e, t=t)
                if self.ikr_control=='deterministic':
                    self.ikr._save_distill_weight(total_loss = total_loss.item(), train_loss = train_loss.item(), kd_loss = kd_loss.item(), epoch = e, t=t)#, iter_num = self.iter_num)

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
        # # Log wandb task t
        # wandb.log({"epoch": e,
        #         "task {} Traing total_loss".format(t): total_total_loss / total_num,
        #         "task {} Traing train_loss".format(t): total_train_loss / total_num, 
        #         "task {} Traing kd_loss".format(t): total_kd_loss / total_num, 
        #         "task {} Traing kd_weight".format(t): total_weight / total_num})

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

    ### lwf eval 코드 가져옴
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, train_loss, kd_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.ref_model(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                ### 수정 필요 (IKR) ikr_kd_use
                total_l, train_l, kd_l,_ = self.criterion(t, outputs, targets.to(self.device), targets_old, epoch = 0, dkd_activation=0, ikr_activation=0) ### 반영 해줘야 하나???
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_l, train_l, kd_l = torch.tensor(total_l), torch.tensor(train_l), torch.tensor(kd_l)
                total_loss += total_l.item() * len(targets)
                train_loss += train_l.item() * len(targets)
                if t > 0:   
                    kd_loss += kd_l.item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

        return total_loss / total_num, train_loss/total_num, kd_loss/total_num , total_acc_taw / total_num, total_acc_tag / total_num
    ###############################

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None, epoch = None, dkd_activation=0, ikr_activation=0):
        """Returns the loss value"""
        total_loss = 0
        train_loss = 0
        kd_loss = 0
        weight = 0

        if ref_outputs is None or ref_features is None:
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            train_loss = nn.CrossEntropyLoss(None)(outputs, targets)
            total_loss = train_loss
            
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).view(-1, 1).to(self.device))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            # Eq. 9: integrated objective


            #================= Loss =================#
            train_loss = loss_ce + loss_mr

        #     kd_loss = loss_dist
        #     weight = self.erf._get_distill_weight(epoch, train_loss.item(), kd_loss.item(),self.distill_percent,self.nepochs, self.cycle_approach)
        #     kd_loss *= weight

        #     total_loss = train_loss + kd_loss

        # return total_loss, train_loss, kd_loss, weight
    
            # Knowledge distillation loss for all previous tasks
            if t > 0 and dkd_activation: ## DKD Activation
                kd_loss = loss_dist
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



    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)


# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
            return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out

import copy
import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
import numpy as np
from ..incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
import matplotlib.pyplot as plt

def grad_replay_strategy_maker(kd_strategy_arr, grad_repeat_weight, grad_repeat_weight_ascent=0, plot=False):
    total_arr = np.array([0.] * 200)
    t = 0
    for e in range(200):
        if kd_strategy_arr[e] > 0:
            total_arr[e] = 0
            t = 0
        else:
            total_arr[e] = t * grad_repeat_weight_ascent + grad_repeat_weight
            t += 1
    if plot:
        plt.figure(figsize=(10, 2))
        plt.plot(total_arr)
        plt.xlabel('epoch')
        plt.ylabel('weight')
        plt.show()
    return total_arr

def kd_strategy_maker(repeat_arr, repeat_period, repeat_delay=0, repeat_period_change=0, repeat_weight_change=1., threshold=0, plot=False, strategy_num=None, total_epoch=210, ):
    """Codes for kd strategy
    Args:
        repeat_arr : 복습의 형태를 정의하는 array
        repeat_period : 복습 간 주기
        repeat_delay : 첫 에폭에 시작하지 않고 약간 delay한 복습
        repeat_period_change : 복습 간 주기의 변화량 (덧셈)
        total_epoch : 총 학습 에폭
        repeat_weight_change : 복습 정도의 증감폭 (곱셈)
        threshold : 너무 작은 kd weight는 0으로 줄이는 것, 효율성 증가
    """
    total_arr = np.array([0.] * total_epoch)
    
    no_weight, give_weight = 0, len(repeat_arr)

    for e in range(total_epoch):
        if e < repeat_delay:
            continue

        if no_weight:  # (N, 0)
            total_arr[e] = 0
            no_weight += -1
            if no_weight == 0:  # (1, 0) -> (0, 0)
                give_weight = len(repeat_arr)
                repeat_period += repeat_period_change

        elif give_weight:  # (0, N)
            total_arr[e] = repeat_arr[-give_weight]
            give_weight += -1
            if give_weight == 0:  # (0, 1) -> (0, 0)
                repeat_arr *= repeat_weight_change
                no_weight = repeat_period
        
    # Plotting
    if plot:
        plt.figure(figsize=(10, 2))
        if strategy_num is None:
            plt.title('KD Epoch Strategy')
        else:
            plt.title(f'KD Epoch Strategy {strategy_num}')
        plt.plot(total_arr)
        plt.xlabel('epoch')
        plt.ylabel('weight')
        plt.show()

    return total_arr

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=5., lamb_mr=1., dist=0.5, K=2, cycle_epoch=10,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False, kd_schedule_type='base', my_strategy=-1,
                 ema_alpha=0.9, grad_replay=1., grad_replay_ascent=0.):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
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
        ##############
        #  추가부분  #
        #############
        self.ema_gradients = None  # loss_dist의 그라디언트 EMA를 저장하는 변수
        self.cur_epoch = 0
        self.previous_t = 0

        # args
        self.cycle_epoch = cycle_epoch
        self.kd_schedule_type = kd_schedule_type
        self.my_strategy = my_strategy

        self.ema_alpha = ema_alpha
        self.grad_replay = grad_replay
        self.grad_replay_ascent = grad_replay_ascent
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
        parser.add_argument('--cycle-epoch', default=10 , type=int, required=False,
            help='Number of "new class embeddings chosen as hard negatives '
                    'for MR loss (default=%(default)s)')
            
        parser.add_argument('--kd-schedule-type', default='base', type=str, required=False,
            help='Number of "new class embeddings chosen as hard negatives '
                    'for MR loss (default=%(default)s)')
        parser.add_argument('--my-strategy', default=-1, type=int, required=False,
            help='Number of "new class embeddings chosen as hard negatives '
                    'for MR loss (default=%(default)s)')
        parser.add_argument('--ema-alpha', default=0.9, type=float, required=False,
            help='Number of "new class embeddings chosen as hard negatives '
                    'for MR loss (default=%(default)s)')
        parser.add_argument('--grad-replay', default=1., type=float, required=False,
            help='Number of "new class embeddings chosen as hard negatives '
                    'for MR loss (default=%(default)s)')
        parser.add_argument('--grad-replay-ascent', default=0., type=float, required=False,
            help='Number of "new class embeddings chosen as hard negatives '
                    'for MR loss (default=%(default)s)')
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

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

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

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        #############################################################
        # 수정부분: self.cur_epoch 관리 부분으로, 기존 코드엔 따로    # 
        # epoch 저장하는 변수가 없어서 추가했으며, t(task)가 넘어가면 #
        # epoch을 0으로 초기화함                                   #
        ##########################################################
        if t != self.previous_t:  # 추가부분
            self.cur_epoch = 0  # 추가부분

        self.previous_t = t  # 추가부분


        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            # Forward previous model
            ref_outputs = None
            ref_features = None
            if t > 0:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)

            self.optimizer.zero_grad()
            loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
            # Backward

            loss.backward()
            self.optimizer.step()

        self.cur_epoch += 1  # 추가부분

    def update_loss_dist_grad_ema(self, current_grad):
        # 그라디언트 EMA를 업데이트합니다.
        if self.loss_dist_grad_ema is None:
            self.loss_dist_grad_ema = current_grad.clone()  # 최초의 그라디언트 EMA는 현재 그라디언트로 초기화
        else:
            self.loss_dist_grad_ema = self.ema_alpha * current_grad.clone() + (1 - self.ema_alpha) * self.loss_dist_grad_ema

    def apply_loss_dist_grad_ema(self, loss_dist):
        # 저장된 그라디언트 EMA를 사용하여 새로운 손실 값의 그라디언트를 설정하고 반환합니다.
        # 새로운 손실 값을 생성합니다. 초기 값은 loss_dist와 동일하게 설정합니다.
        new_loss_dist = torch.tensor(loss_dist.item(), requires_grad=True).to('cuda')
        # 새로운 손실 값에 그라디언트를 설정합니다.
        new_loss_dist.grad = self.loss_dist_grad_ema.clone()

        return new_loss_dist

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None:
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
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
                                                                     torch.ones(hard_num * self.K).to(self.device))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            # Eq. 9: integrated objective

            #########################################################
            # 수정부분: 이런 식으로 cur_epoch을 받아서 loss에 dist 추가#
            #########################################################
            if self.my_strategy == -1:
                if self.kd_schedule_type == 'base':
                    loss = loss_dist + loss_ce + loss_mr
                elif self.kd_schedule_type == 'cycle':
                    if self.cur_epoch % self.cycle_epoch == 0: loss = loss_dist + loss_ce + loss_mr
                    else: loss = loss_ce + loss_mr
                elif self.kd_schedule_type == 'first':
                    if self.cur_epoch < self.cycle_epoch: loss = loss_dist + loss_ce + loss_mr
                    else: loss = loss_ce + loss_mr
                elif self.kd_schedule_type == 'last':
                    if self.cur_epoch >= 200 - self.cycle_epoch: loss = loss_dist + loss_ce + loss_mr
                    else: loss = loss_ce + loss_mr
                elif self.kd_schedule_type == 'middle':
                    if 100 - self.cycle_epoch // 2 < self.cur_epoch < 100 + self.cycle_epoch // 2: loss = loss_dist + loss_ce + loss_mr
                    else: loss = loss_ce + loss_mr
                elif self.kd_schedule_type == 'firstlast':
                    if self.cur_epoch < self.cycle_epoch // 2 or self.cur_epoch > 200 - self.cycle_epoch // 2: loss = loss_dist + loss_ce + loss_mr
                    else: loss = loss_ce + loss_mr
            else:
                if self.my_strategy == 0:
                    arr = np.arange(0, 3, 0.4)
                    arr = np.exp(arr)
                    arr =  - (arr* 2 / max(arr) - 2)
                    total_arr = kd_strategy_maker(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.)
                    loss = total_arr[self.cur_epoch] * loss_dist + loss_ce + loss_mr
                    
                elif self.my_strategy == 1:
                    arr = np.arange(0,3, 0.4)
                    arr = np.concatenate((np.zeros(5), 1.5*np.log(arr + 1)))

                    total_arr = kd_strategy_maker(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.)
                    loss = total_arr[self.cur_epoch] * loss_dist + loss_ce + loss_mr

                # grad 부분
                elif self.my_strategy == 2:
                    arr = np.arange(0, 3, 0.4)
                    arr = np.exp(arr)
                    arr =  - (arr* 2 / max(arr) - 2)
                    total_arr = kd_strategy_maker(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.)
                    grad_replay_arr = grad_replay_strategy_maker(total_arr, self.grad_replay, self.grad_replay_ascent)

                    if total_arr[self.cur_epoch] == 0:
                        if self.ema_gradients:
                            for name, param in self.model.named_parameters():
                                if self.ema_gradients[name] is not None: param.grad = grad_replay_arr[self.cur_epoch] * self.ema_gradients[name]
                                else: param.grad = None
                        loss = loss_ce + loss_mr  

                    else:
                        self.optimizer.zero_grad()
                        loss_dist.backward(retain_graph=True)  # loss_dist gradient 계산
                        current_gradients = dict()

                        for name, param in self.model.named_parameters():
                            if param.grad is not None: current_gradients[name] = param.grad.clone().detach()
                            else: current_gradients[name] = None

                        if self.ema_gradients:
                            for name, param in self.model.named_parameters():
                                if param.grad is not None: self.ema_gradients[name] = self.ema_alpha * self.ema_gradients[name] + (1 - self.ema_alpha) * current_gradients[name]  # EMA 저장
                                else: self.ema_gradients[name] = None
                                
                        else:  # ema_gradients == None일 땐 초기화
                            self.ema_gradients = current_gradients

                        loss = loss_dist * total_arr[self.cur_epoch] + loss_ce + loss_mr

                elif self.my_strategy == 3:  # temp
                    arr = np.arange(0, 3, 0.4)
                    arr = np.exp(arr)
                    arr =  - (arr* 2 / max(arr) - 2)
                    total_arr = kd_strategy_maker(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.)
                    
                    if total_arr[self.cur_epoch] == 0:
                        if self.ema_gradients:
                            for name, param in self.model.named_parameters():
                                if self.ema_gradients[name] is not None: param.grad = 0.5 * self.ema_gradients[name]
                                else: param.grad = None
                        loss = loss_ce + loss_mr  

                    else:
                        self.optimizer.zero_grad()
                        temp_loss = loss_dist * total_arr[self.cur_epoch]
                        temp_loss.backward(retain_graph=True)  # loss_dist gradient 계산
                        current_gradients = dict()

                        for name, param in self.model.named_parameters():
                            if param.grad is not None: current_gradients[name] = param.grad.clone().detach()
                            else: current_gradients[name] = None

                        if self.ema_gradients:
                            for name, param in self.model.named_parameters():
                                if param.grad is not None: self.ema_gradients[name] = self.ema_alpha * self.ema_gradients[name] + (1 - self.ema_alpha) * current_gradients[name]  # EMA 저장
                                else: self.ema_gradients[name] = None
                                
                        else:  # ema_gradients == None일 땐 초기화
                            self.ema_gradients = current_gradients

                        loss = temp_loss + loss_ce + loss_mr

                elif self.my_strategy == 4:  # mine
                    arr = np.arange(0, 3, 0.4)
                    arr = np.exp(arr)
                    arr =  - (arr* 2 / max(arr) - 2)
                    total_arr = kd_strategy_maker(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.1)
                    loss = total_arr[self.cur_epoch] * loss_dist + loss_ce + loss_mr

                elif self.my_strategy == 5:  # mine
                    arr = np.arange(0, 3, 0.4)
                    arr = np.exp(arr)
                    arr =  - (arr* 2 / max(arr) - 2)
                    total_arr = kd_strategy_maker(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.2)
                    loss = total_arr[self.cur_epoch] * loss_dist + loss_ce + loss_mr

        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train 10700 + 1250
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

import numpy as np
import pandas as pd
import torch

class IKR():
    def __init__(self, pk_approach, ikr_switch, recycle_percent, total_epochs,  dkd_control, dkd_switch, distill_percent, m):

        self.pk_approach      = pk_approach
        self.ikr_switch       = ikr_switch
        self.recycle_percent  = recycle_percent
        self.m                = m
        self.total_epochs     = total_epochs

        self.dkd_control      = dkd_control
        self.dkd_switch       = dkd_switch
        self.distill_percent  = distill_percent

        self.total_loss = [0]*total_epochs
        self.train_loss = [0]*total_epochs
        self.kd_loss = [0]*total_epochs

        self.valid_total_loss = [0]*total_epochs
        self.valid_train_loss = [0]*total_epochs
        self.valid_kd_loss = [0]*total_epochs
        self.valid_taw_accuracies = [0]*total_epochs
        self.valid_tag_accuracies = [0]*total_epochs

        self.current_task = 0

        self.ikr_switch_array = [0]*total_epochs
        self.iter_num = 0

    def _get_distill_weight(self, epoch, dkd_activation):
        ikr_m_tensor = torch.tensor(self.m,requires_grad =True)
        # return self.m * self._make_recycling_weight(epoch,dkd_activation)
        # return ikr_m_tensor * self._make_recycling_weight(epoch, dkd_activation)
        ikr_m_tensor = torch.tensor(self.m, requires_grad=True)
        recycling_weight = self._make_recycling_weight(epoch, dkd_activation)
        return ikr_m_tensor * recycling_weight  # 역전파 그래프에 포함된 weight 반환
    
    def _make_recycling_weight(self,epoch,dkd_activation):
        return self._switch_function(epoch,dkd_activation) * self._pk_function(epoch)

    def _save_distill_weight(self, total_loss = None, train_loss = None, kd_loss = None, valid_total_loss = None, valid_train_loss = None, valid_kd_loss = None, valid_taw_accuracie = None, valid_tag_accuracies = None,  epoch = 0):#, iter_num=0):
        # save the training loss
        if total_loss is not None:
            self.total_loss[epoch] += total_loss
        if train_loss is not None:
            self.train_loss[epoch] += train_loss
        if kd_loss is not None:
            self.kd_loss[epoch] += kd_loss

        # save the validation loss
        if valid_total_loss is not None:
            self.valid_total_loss[epoch] += valid_total_loss
        if valid_train_loss is not None:
            self.valid_train_loss[epoch] += valid_train_loss
        if valid_kd_loss is not None:
            self.valid_kd_loss[epoch] += valid_kd_loss

        # save the validation accuracies
        if valid_taw_accuracie is not None:
            self.valid_taw_accuracies[epoch] += valid_taw_accuracie
        if valid_tag_accuracies is not None:
            self.valid_tag_accuracies[epoch] += valid_tag_accuracies

        # save total iteration number in trn_loader
        #self.iter_num = iter_num

    def _reset_save(self, task):
        if self.current_task != task:
            self.current_task = task
            self.total_loss = [0]*self.total_epochs
            self.train_loss = [0]*self.total_epochs
            self.kd_loss = [0]*self.total_epochs

            self.valid_total_loss = [0]*self.total_epochs
            self.valid_train_loss = [0]*self.total_epochs
            self.valid_kd_loss = [0]*self.total_epochs
            self.valid_taw_accuracies = [0]*self.total_epochs
            self.valid_tag_accuracies = [0]*self.total_epochs
    
    def _pk_function(self,epoch):
        if self.dkd_switch == 'end':
            return 0
        elif self.dkd_switch == 'cycle':
            if 'average' == self.pk_approach:
                kd_losses_to_process = self.kd_loss[:epoch][::-1]
                kd_losses_to_process = [loss for loss in kd_losses_to_process if loss > 0]
                window_size = 10 ##### 활성화된 KD loss의 지난 10에포크만 고려해 average 취함
                if len(kd_losses_to_process)>=10:
                    avg_kd_loss = sum(kd_losses_to_process[:window_size])/window_size
                else:
                    avg_kd_loss = sum(kd_losses_to_process)/len(kd_losses_to_process)
                return avg_kd_loss / self.iter_num
            elif 'last' == self.pk_approach:
                kd_losses_to_process = self.kd_loss[:epoch][::-1]
                kd_losses_to_process = [loss for loss in kd_losses_to_process if loss > 0]
                return kd_losses_to_process[0] / self.iter_num
            elif 'ema' == self.pk_approach:
                kd_losses_to_process = self.kd_loss[:epoch][::-1]
                kd_losses_to_process = [loss for loss in kd_losses_to_process if loss > 0]
                if len(kd_losses_to_process)<10:
                    return kd_losses_to_process[0]
                else:
                    kd_values_ori = pd.DataFrame({'kd_values':kd_losses_to_process[::-1]})
                    ema = kd_values_ori.ewm(span=10).mean()
                    ema = ema['kd_values'].values
                    return ema[-1] / self.iter_num

        else: ## first, mid, first_end, ten_to_ten
            if 'average' == self.pk_approach:
                kd_losses_to_process = self.kd_loss[:epoch][::-1]
                avg_kd_loss = 0
                cnt = 0
                while kd_losses_to_process:
                    if kd_losses_to_process[0]==0:
                        if avg_kd_loss == 0:
                            kd_losses_to_process.pop(0)
                        else:
                            break
                    else:
                        avg_kd_loss+=kd_losses_to_process.pop(0)
                        cnt +=1
                avg_kd_loss /= cnt
                return avg_kd_loss / self.iter_num
            elif 'last' == self.pk_approach:
                kd_losses_to_process = self.kd_loss[:epoch][::-1]
                kd_losses_to_process = [loss for loss in kd_losses_to_process if loss > 0]
                return kd_losses_to_process[0] / self.iter_num
            elif 'ema' == self.pk_approach:
                kd_losses_to_process = self.kd_loss[:epoch][::-1]
                required_kd_losses = []
                avg_kd_loss = 0
                while kd_losses_to_process:
                    if kd_losses_to_process[0]==0:
                        if avg_kd_loss == 0:
                            kd_losses_to_process.pop(0)
                        else:
                            break
                    else:
                        avg_kd_loss+=kd_losses_to_process[0]
                        required_kd_losses.append(kd_losses_to_process.pop(0))
                if len(required_kd_losses)<10:
                    return required_kd_losses[0] / self.iter_num
                else:
                    kd_values_ori = pd.DataFrame({'kd_values':required_kd_losses[::-1]})
                    ema = kd_values_ori.ewm(span=10).mean()
                    ema = ema['kd_values'].values
                    return ema[-1] / self.iter_num
                
    def _dkd_adaptive_update(self, nai_s, ndi_s,Ni_s,li_s,gi_s,thresholds,dkd_switch_array_update):
        self.nai_s = nai_s
        self.ndi_s = ndi_s
        self.Ni_s = Ni_s
        self.dkd_switch_array_update = dkd_switch_array_update

    def _dkd_adaptive_switch_array(self):
        self.dkd_switch_arr = []
        for i in range(len(self.Ni_s)):
            self.dkd_switch_arr.extend([1]*self.nai_s[i])
            self.dkd_switch_arr.extend([0]*self.ndi_s[i])

    def _dkd_deterministic_switch_array(self):
        if self.dkd_switch =='all':
            self.dkd_switch_arr = [1]*self.total_epochs
            self.ikr_switch_arr = [0]*self.total_epochs
        elif self.dkd_switch =='end':
            distill_epochs=int(self.distill_percent*self.total_epochs)
            self.dkd_switch_arr = [0]*(1-distill_epochs) + [1]*distill_epochs
            self.ikr_switch_arr = [0]*self.total_epochs
        elif self.dkd_switch == 'cycle':
            self.dkd_switch_arr = []
            for epoch in range(self.total_epochs):
                    if self.distill_percent == 0.2:
                        if epoch % 5 == 0:
                            self.dkd_switch_arr.append(1)
                        else:
                            self.dkd_switch_arr.append(0)
        elif self.dkd_switch == 'first':
            distill_epochs = int(self.distill_percent * self.total_epochs)
            self.dkd_switch_arr = [1 if i<distill_epochs else 0 for i in range(self.total_epochs) ]
        elif self.dkd_switch == 'mid':
            distill_epochs = int(self.distill_percent * self.total_epochs)
            # self.dkd_switch_arr = [1 if (((i >= (self.total_epochs // 2 - distill_epochs // 2)) and (i < (self.total_epochs // 2 + distill_epochs else 0 // 2)))) else 0 for i in range(self.total_epochs)]
            self.dkd_switch_arr = [
                1 if ((i >= (self.total_epochs // 2 - distill_epochs // 2)) and (i < (self.total_epochs // 2 + distill_epochs // 2))) else 0
                for i in range(self.total_epochs)
            ]
        elif self.dkd_switch == 'end':
            distill_epochs = int(self.distill_percent * self.total_epochs)
            self.dkd_switch_arr = [1 if i >= (self.total_epochs - self.distill_epoch) else 0 for i in range(self.total_epochs)]
        elif self.dkd_switch == 'first_end':
            distill_epochs = int(self.distill_percent * self.total_epochs)
            self.dkd_switch_arr = [1 if (i < distill_epochs // 2) or (i>= self.total_epochs - distill_epochs//2) else 0 for i in range(self.total_epochs)]
        elif self.dkd_switch == 'ten_to_ten':
            distill_epochs = int(self.distill_percent * self.total_epochs)
            n_gap = distill_epochs // 10
            gap = int(self.total_epochs * (1 - self.distill_percent) / n_gap)
            ten_to_ten_list = []
            for point in range(0,200, 10+ gap):
                ten_to_ten_list.extend(range(point, point+10))
            self.dkd_switch_arr = [1 if i in self.ten_to_ten_list else 0 for i in range(self.total_epochs)]



    def _switch_function(self,epoch,dkd_activation):
        if self.current_task ==0:
            return 0
        elif dkd_activation == 1:
            return 0
        else:
            if self.dkd_control ==  'deterministic':
                if self.dkd_switch == 'ten_to_ten':
                    ikr_switch_arr = np.array([0]*50)

                    ikr_switch_arr[13:23]=1
                    ikr_switch_arr[25:35]=1
                    ikr_switch_arr[37:47]=1

                    ikr_switch_arr = np.concatenate([ikr_switch_arr,ikr_switch_arr,ikr_switch_arr,ikr_switch_arr])
                    return ikr_switch_arr[epoch]
                elif self.dkd_switch == 'cycle':
                    ikr_switch_arr = [0,1,0,1,1]*40
                    return ikr_switch_arr[epoch]
                elif self.dkd_switch == 'first':
                    ikr_switch_arr = [0]*40+[1]*120+[0]*40
                    return ikr_switch_arr[epoch]
                elif self.dkd_switch == 'mid':
                    ikr_switch_arr = [0]*120+[1]*80
                    return ikr_switch_arr[epoch]
                elif self.dkd_switch == 'end':
                    return 0
                elif self.dkd_switch == 'first_end':
                    ikr_switch_arr = [0]*20+[1]*60+[0]*40+[1]*60+[0]*20
                    return ikr_switch_arr[epoch]
                
            elif self.dkd_control ==  'adaptive':
                ikr_switch_arr =[]
                for i in range(len(self.ndi_s)):
                    if self.ndi_s[i]>30:
                        zero_nums = self.ndi_s[i]-20
                        num_zero_per_period = int(zero_nums/3)
                        remaining_zeros = zero_nums%3
                    else:
                        zero_nums = self.ndi_s[i]-10
                        num_zero_per_period = int(zero_nums/2)
                        remaining_zeros = zero_nums%2
                        
                    
                    
                    if self.ndi_s[i]>30:
                        ikr_switch_arr += [0]*self.nai_s[i] + ([0]*num_zero_per_period + [1]*10)*2 + [0]*(num_zero_per_period+remaining_zeros)
                    else:
                        ikr_switch_arr += [0]*self.nai_s[i] + ([0]*num_zero_per_period + [1]*10)*1 + [0]*(num_zero_per_period+remaining_zeros)
                    
                return ikr_switch_arr[epoch]
            
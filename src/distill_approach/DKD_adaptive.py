import numpy as np

class DKD_adaptive():
    def __init__(self, dkd_switch, dkd_shape,total_epochs, m = 2):
        self.dkd_shape = dkd_shape
        self.dkd_switch = dkd_switch
        self.m = m
        self.total_epochs = total_epochs

        self.total_loss = [0]*total_epochs
        self.train_loss = [0]*total_epochs
        self.kd_loss = [0]*total_epochs

        self.valid_total_loss = [0]*total_epochs
        self.valid_train_loss = [0]*total_epochs
        self.valid_kd_loss = [0]*total_epochs
        self.valid_taw_accuracies = [0]*total_epochs
        self.valid_tag_accuracies = [0]*total_epochs

        self.current_task = 0

        ## swtich_st
        self.nai_s=[10]
        self.ndi_s=[40]
        self.Ni_s=[50]
        self.li_s=[]
        self.gi_s=[]
        self.na_max_add=5
        self.na_min=5
        self.nd_min=20
        self.thresholds=[]
        self.quantile_threshold = 0.5
        ###

    def _get_distill_weight(self, epoch):
        return self.m * self._make_adaptive_distll_weight(epoch)
     
    def _make_adaptive_distll_weight(self, epoch):
        if self._switch_function(epoch) == 1:
            return self._switch_function(epoch) * self._shape_function(epoch)
        else:
            return 0

    def _save_distill_weight(self, total_loss = None, train_loss = None, kd_loss = None, valid_total_loss = None, valid_train_loss = None, valid_kd_loss = None, valid_taw_accuracie = None, valid_tag_accuracies = None,  epoch = 0,t=0):
        # update self.t
        self.current_task = t
        
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

            ## swtich_st
            self.nai_s=[10]
            self.ndi_s=[40]
            self.Ni_s=[50]
            self.li_s=[]
            self.gi_s=[]
            self.na_max_add=5
            self.na_min=5
            self.nd_min=20
            self.thresholds=[]
            self.quantile_threshold = 0.5
            ###
    
    def _nai_ndi_update(self,epoch): ## 다음 학습을 위해 nai,ndi 계산
        last_nai_kds=np.array(self.kd_loss[int(self.Ni_s[-1]-self.ndi_s[-1]-self.nai_s[-1]):int(self.Ni_s[-1]-self.ndi_s[-1])])
        # l_i와 g_i를 계산하기 위한 thresholds 업데이트
        self.thresholds.append(np.quantile(last_nai_kds,self.quantile_threshold)) ## 사분위수 threshold로 지정
        # if min(self.kd_loss[int(self.Ni_s[-1]-self.ndi_s[-1]-self.nai_s[-1]):int(self.Ni_s[-1])])<self.thresholds[-1]:
        #     self.thresholds.append(self.thresholds[-1]-0.05)
        # else:
        #     self.thresholds.append(self.thresholds[-1])

        #l_i 계산
        l_i=sum(last_nai_kds<=self.thresholds[-1])
        self.li_s.append(l_i)
        #g_i 계산
        g_i=int(self.nai_s[-1])-l_i
        self.gi_s.append(g_i)
        

        # nai+1과 ndi+1 계산
        if self.dkd_switch=='ada_switch_2':
            na_delta=g_i-l_i
            nd_delta=l_i-g_i
            new_na=self.nai_s[-1]+na_delta
            new_nd=self.ndi_s[-1]+nd_delta
        elif self.dkd_switch=='ada_switch_3':
            if g_i>=l_i:
                sign=1
            else:
                sign=-1
            new_na=self.nai_s[-1] * (1+((-1)**(1-max(0,sign)))*(g_i/self.nai_s[-1]))
            new_nd=self.ndi_s[-1] * (1+((-1)**(1-max(0,-sign)))*(l_i/self.nai_s[-1]))
        elif self.dkd_switch=='ada_switch_4':
            if g_i>=l_i:
                sign=1
            else:
                sign=-1
            new_na=self.nai_s[-1] * (1+((-1)**(1-max(0,sign)))*(1+np.e**(g_i/self.nai_s[-1])))
            new_nd=self.ndi_s[-1] * (1+((-1)**(1-max(0,-sign)))*(1+np.e**(l_i/self.nai_s[-1])))
        elif self.dkd_switch=='ada_switch_5':
            if g_i>=l_i:
                sign=1
            else:
                sign=-1
            new_na=self.nai_s[-1] * (1+((-1)**(1-max(0,sign)))*(1+np.log(g_i/self.nai_s[-1])))
            new_nd=self.ndi_s[-1] * (1+((-1)**(1-max(0,-sign)))*(1+np.log(l_i/self.nai_s[-1])))

        try: ## 계산된 new_na가 정상적일 때
            new_na = int(new_na)
        except: ## 계산된 new_na가 비정상적일 때, INF
            new_na = self.nai_s[-1]+self.na_max_add
        try: ## 계산된 new_nd가 정상적일 때
            new_nd = int(new_nd)
        except: ## 계산된 new_nd가 비정상적일 때, INF
            new_nd = self.ndi_s[-1]+self.na_max_add
        
        ## 클리핑
        if new_na>(self.nai_s[-1]+self.na_max_add):
            self.nai_s.append(self.nai_s[-1]+self.na_max_add)
        elif new_na<self.na_min:
            self.nai_s.append(self.na_min)
        else:
            self.nai_s.append(new_na)
        if new_nd>(self.ndi_s[-1]+self.na_max_add):
            self.ndi_s.append(self.ndi_s[-1]+self.na_max_add)
        elif new_nd<self.nd_min:
            self.ndi_s.append(self.nd_min)
        else:
            self.ndi_s.append(new_nd)
        
        self.Ni_s.append(self.Ni_s[-1]+self.nai_s[-1]+self.ndi_s[-1])


    def _switch_function(self,epoch):
        if self.dkd_switch =='all': ### ada_shape_1
            return 1
        else:  ### ada_shape_2 ~ 5
            if self.current_task !=0:
                # 첫 10에포크 활성화
                if epoch<self.nai_s[0]:
                    return 1
                # 첫 11~50에포크 비활성화
                elif epoch<self.Ni_s[0]:
                    return 0
                # 그 이후
                else:
                    if self.dkd_switch =='ada_switch_1':
                        remaining_kd_losses = [loss for loss in self.kd_loss if loss>0]
                        threshold = np.quantile(remaining_kd_losses , self.quantile_threshold) ## 제1사분위수를 threshold로 지정
                        if remaining_kd_losses[-1]>=threshold:
                            return 1
                        else:
                            # self.thresholds.append(self.thresholds[-1]-0.05)
                            return 0
                    else:
                        if epoch==self.Ni_s[-1]:    ## nai_ndi 업데이트
                            self._nai_ndi_update(epoch)
                        if (self.Ni_s[-1]-self.ndi_s[-1]-self.nai_s[-1])<=epoch<(self.Ni_s[-1]-self.ndi_s[-1]):
                            return 1
                        else:
                            return 0
            else: # self.current_task ==0
                return 0
    
    def _shape_function(self,epoch):
        if self.dkd_shape == 'ada_shape_1':
            if self.total_loss[epoch] == 0:
                return 1
            else:
                return self.kd_loss[epoch] / self.total_loss[epoch]
        
        elif self.dkd_shape == 'ada_shape_2':
            if epoch < 2:
                return 1
            elif self.valid_kd_loss[epoch-2]==0:
                return 1
            else:
                return self.valid_kd_loss[epoch-1] / self.valid_kd_loss[epoch-2]
        
        elif self.dkd_shape == 'ada_shape_3':
            if epoch < 2:
                return 1
            elif self.valid_train_loss[epoch-2]==0:
                return 1
            else:
                return self.valid_train_loss[epoch-1] / self.valid_train_loss[epoch-2] 
        
        elif self.dkd_shape == 'ada_shape_4':
            if epoch < 2:
                return 1
            elif self.valid_total_loss[epoch-2]==0:
                return 1
            else:
                return self.valid_total_loss[epoch-1] / self.valid_total_loss[epoch-2] 
        
        elif self.dkd_shape == 'ada_shape_5':
            if epoch < 1:
                return 1
            elif self.valid_tag_accuracies[epoch-1]==0:
                return 1
            else:
                return 1 - self.valid_taw_accuracies[epoch-1] / self.valid_tag_accuracies[epoch-1]
        else: ## one
            return 1



    

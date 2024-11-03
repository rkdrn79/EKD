import math

class DKD_deterministic():
    def __init__(self, dkd_switch, dkd_shape, total_epochs,distill_percent, m = 2):
        self.dkd_switch = dkd_switch
        self.dkd_shape = dkd_shape
        self.m = m
        self.total_epochs = total_epochs
        self.distill_percent = distill_percent
        self.num_distill_epoch = int(self.total_epochs * self.distill_percent)

        self.total_loss = [0]*total_epochs
        self.train_loss = [0]*total_epochs
        self.kd_loss = [0]*total_epochs

        self.valid_total_loss = [0]*total_epochs
        self.valid_train_loss = [0]*total_epochs
        self.valid_kd_loss = [0]*total_epochs
        self.valid_taw_accuracies = [0]*total_epochs
        self.valid_tag_accuracies = [0]*total_epochs

        self.current_task = 0


    def _get_distill_weight(self,epoch):
        return self.m * self._make_deterministic_distill_weight(epoch)

    def _make_deterministic_distill_weight(self, epoch):
        if self._switch_function(epoch) == 1:
            return self._switch_function(epoch) * self._shape_function(epoch)
        else:
            return 0

    def _save_distill_weight(self, total_loss = None, train_loss = None, kd_loss = None, valid_total_loss = None, valid_train_loss = None, valid_kd_loss = None, valid_taw_accuracie = None, valid_tag_accuracies = None,  epoch = 0,t=0):
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

    def _switch_function(self,epoch):
        if self.dkd_switch == 'none': # finetune
            return 0
        elif self.dkd_switch == 'all':
            return 1
        elif self.dkd_switch == 'cycle':
            if self.distill_percent < 0.5:
                if self.distill_percent == 0.2:
                    if epoch % 5 == 0:
                        return 1
                    else:
                        return 0
                elif self.distill_percent == 0.4:
                    if epoch % 5 in [1,3]:
                        return 1
                    else:
                        return 0
            else:
                if self.distill_percent == 0.8:
                    if epoch %  5 != 0:
                        return 1
                    else:
                        return 0
                elif self.distill_percent == 0.6:
                    if epoch % 5 in [0,2,4]:
                        return 1
                    else:
                        return 0
        elif self.dkd_switch == 'first':
            if epoch < self.num_distill_epoch :
                return 1
            else:
                return 0
        elif self.dkd_switch == 'mid':
            if (epoch >= self.total_epochs // 2 - self.num_distill_epoch // 2) and (epoch < self.total_epochs // 2 + self.num_distill_epoch // 2):
                return 1
            else:
                return 0
        elif self.dkd_switch == 'end':
            if epoch >= self.total_epochs - self.num_distill_epoch:
                return 1
            else:
                return 0
        elif self.dkd_switch == 'first_end':
            if (epoch < self.num_distill_epoch // 2) or (epoch>= self.total_epochs - self.num_distill_epoch//2):
                return 1
            else:
                return 0
        elif self.dkd_switch == 'ten_to_ten':
            self.n_gap = self.num_distill_epoch // 10
            self.gap = int(self.total_epochs * (1 - self.distill_percent) / self.n_gap)
            self.ten_to_ten_list = []
            for point in range(0,200, 10+ self.gap):
                self.ten_to_ten_list.extend(range(point, point+10))
            if epoch in self.ten_to_ten_list:
                return 1
            else:
                return 0
        elif self.dkd_approach == 'custom':
            return self._make_switch_array()[epoch]

    def _shape_function(self,epoch):
        if self.dkd_shape == 'convexed_increase':
            if self.dkd_switch == 'cycle':
                return 1 * math.sin(math.pi * (epoch) / (2 * self.total_epochs)+ 1.5 * math.pi) + 1
            elif self.dkd_switch == 'first_end':
                if epoch > self.total_epochs/2:
                    return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent/2)) / (self.total_epochs * self.distill_percent)+ 1.5 * math.pi)+ 1
                else:
                    return 1 * math.sin(math.pi * (epoch) / (self.total_epochs * self.distill_percent)+ 1.5 * math.pi) + 1
            elif self.dkd_switch == 'first':
                # epoch = x
                return 1 * math.sin(math.pi * (epoch) / (2 * self.total_epochs * self.distill_percent)+ 1.5 * math.pi) + 1
            elif self.dkd_switch == 'mid':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent) / 2) / (2 * self.total_epochs * self.distill_percent) + 1.5 * math.pi) + 1
            elif self.dkd_switch == 'end':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent)) / (2 * self.total_epochs * self.distill_percent)+ 1.5 * math.pi)+ 1
            elif self.dkd_switch == 'ten_to_ten':
                n_gap = int (self.total_epochs * self.distill_percent) // 10
                gap = int (self.total_epochs * (1-self.distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return 1 * math.sin(math.pi * (epoch - starting_point) / (2 * (10)) + 1.5 * math.pi) + 1
            else:
                return 1
            
        elif self.dkd_shape == 'convexed_decrease':
            if self.dkd_switch == 'cycle':
                return 1 * math.sin(math.pi * (epoch) / (2 * self.total_epochs)+ math.pi) + 1
            elif self.dkd_switch == 'first_end':
                if epoch > self.total_epochs/2:
                    return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent/2)) / (self.total_epochs * self.distill_percent)+ math.pi)+ 1
                else:
                    return 1 * math.sin(math.pi * (epoch) / (self.total_epochs * self.distill_percent)+ math.pi) + 1
            elif self.dkd_switch == 'first':
                # epoch = x
                return 1 * math.sin(math.pi * (epoch) / (2 * self.total_epochs * self.distill_percent)+ math.pi) + 1
            elif self.dkd_switch == 'mid':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent) / 2) / (2 * self.total_epochs * self.distill_percent)+ math.pi) + 1
            elif self.dkd_switch == 'end':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent)) / (2 * self.total_epochs * self.distill_percent)+ math.pi)+ 1
            elif self.dkd_switch == 'ten_to_ten':
                n_gap = int (self.total_epochs * self.distill_percent) // 10
                gap = int (self.total_epochs * (1-self.distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return 1 * math.sin(math.pi * (epoch-starting_point) / (2 * (10)) + math.pi) + 1
            else:
                return 1
            
        elif self.dkd_shape == 'concaved_increase':
            if self.dkd_switch == 'cycle':
                return 1 * math.sin(math.pi * epoch / (2 * self.total_epochs))
            elif self.dkd_switch == 'first_end':
                if epoch > self.total_epochs/2:
                    return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent/2)) / (self.total_epochs * self.distill_percent))
                else:
                    return 1 * math.sin(math.pi * epoch / (self.total_epochs * self.distill_percent))
            elif self.dkd_switch == 'first':
                # epoch = x
                return 1 * math.sin(math.pi * epoch / (2 * self.total_epochs * self.distill_percent))
            elif self.dkd_switch == 'mid':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent) / 2) / (2 * self.total_epochs * self.distill_percent)) 
            elif self.dkd_switch == 'end':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent)) / (2 * self.total_epochs * self.distill_percent))
            elif self.dkd_switch == 'ten_to_ten':
                n_gap = int (self.total_epochs * self.distill_percent) // 10
                gap = int (self.total_epochs * (1-self.distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return math.sin(math.pi * (epoch - starting_point) / (2 * (10)))
            else:
                return 1 * 1
            
        elif self.dkd_shape == 'concaved_decrease':
            if self.dkd_switch == 'cycle':
                return 1 * math.sin(math.pi * (epoch) / (2 * self.total_epochs)+ 0.5 * math.pi)
            elif self.dkd_switch == 'first_end':
                if epoch > self.total_epochs/2:
                    return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent/2)) / (self.total_epochs * self.distill_percent) + math.pi/2)
                else:
                    return 1 * math.sin(math.pi * (epoch) / (self.total_epochs * self.distill_percent) + 0.5 * math.pi)
            elif self.dkd_switch == 'first':
                # epoch = x
                return 1 * math.sin(math.pi * (epoch) / (2 * self.total_epochs * self.distill_percent)+ 0.5 * math.pi)
            elif self.dkd_switch == 'mid':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent) / 2) / (2 * self.total_epochs * self.distill_percent)+ math.pi/2) 
            elif self.dkd_switch == 'end':
                return 1 * math.sin(math.pi * (epoch - self.total_epochs * (1 - self.distill_percent)) / (2 * self.total_epochs * self.distill_percent)+ math.pi/2)
            elif self.dkd_switch == 'ten_to_ten':
                n_gap = int (self.total_epochs * self.distill_percent) // 10
                gap = int (self.total_epochs * (1-self.distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return math.sin(math.pi * (epoch - starting_point) / (2 * (10)) + 0.5 * math.pi)
            else:
                return 1
            
        elif self.dkd_shape == 'linear_increase':
            if self.dkd_switch == 'cycle':
                return 1 /self.total_epochs * epoch
            elif self.dkd_switch == 'first_end':
                if epoch > self.total_epochs/2:
                    return 1 / (self.total_epochs * self.distill_percent/2) * (epoch - self.total_epochs * (1 - self.distill_percent))
                else:
                    return 1 / (self.total_epochs * self.distill_percent/2) * epoch
            elif self.dkd_switch == 'first':
                # epoch = x
                return 1 / (self.total_epochs * self.distill_percent) * epoch
            elif self.dkd_switch == 'mid':
                return 1 / (self.total_epochs * self.distill_percent) * (epoch - self.total_epochs * (1 - self.distill_percent) / 2)
            elif self.dkd_switch == 'end':
                return 1 / (self.total_epochs * self.distill_percent) * (epoch - self.total_epochs * (1 - self.distill_percent))
            elif self.dkd_switch == 'ten_to_ten':
                n_gap = int (self.total_epochs * self.distill_percent) // 10
                gap = int (self.total_epochs * (1-self.distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return 1 / 10 * (epoch - starting_point)
            else:
                return 1
        elif self.dkd_shape == 'linear_decrease':
            if self.dkd_switch == 'cycle':
                return -1 * 1 / (self.total_epochs) * epoch + 1
            elif self.dkd_switch == 'first_end':
                if epoch > self.total_epochs/2:
                    return -1 * 1 / (self.total_epochs * self.distill_percent/2) * epoch + 1
                else:
                    return -1 * 1 / (self.total_epochs * self.distill_percent/2) * (epoch - self.total_epochs * (1 - self.distill_percent/2)) + 1
            elif self.dkd_switch == 'first':
                # epoch = x
                return -1 * 1 / (self.total_epochs * self.distill_percent) * epoch + 1
            elif self.dkd_switch == 'mid':
                return -1 * 1 / (self.total_epochs * self.distill_percent) * (epoch - self.total_epochs * (1 - self.distill_percent) / 2) + 1
            elif self.dkd_switch == 'end':
                return -1 * 1 / (self.total_epochs * self.distill_percent) * (epoch - self.total_epochs * (1 - self.distill_percent)) + 1
            elif self.dkd_switch == 'ten_to_ten':
                n_gap = int (self.total_epochs * self.distill_percent) // 10
                gap = int (self.total_epochs * (1-self.distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return -1 * 1 / 10 * (epoch - starting_point) + 1
            else:
                return 1
        else:
            return 1

    
    def _make_switch_array(self, repeat_num=5, repeat_num_decrease=-1, repeat_period=80, repeat_delay=5, repeat_period_change=-10):
        """
        Codes for kd strategy
        Args:
            total_epoch : 총 학습 에폭 = train_epochs
            repeat_num : 한 복습 주기당 몇 번의 복습을 할 것인지
            repeat_num_decrease : 복습 주기가 지날 때마다 repeat_num에 더해질 값
            repeat_period : 복습 간 주기
            repeat_delay : 첫 에폭에 시작하지 않고 약간 delay한 복습
            repeat_period_change : 복습 간 주기의 변화량 (덧셈)
        """
        
        switch_array = [False] * self.total_epochs
        repeat_num = repeat_num
        for i in range(repeat_delay, self.total_epochs, repeat_period):
            for j in range(repeat_num):
                switch_array[i + j] = True
            repeat_num += repeat_num_decrease
            repeat_period += repeat_period_change
        return switch_array
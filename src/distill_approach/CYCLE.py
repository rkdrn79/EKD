class CYCLE():
    def __init__(self, cycle_approach, train_epochs = 200, distill_percent = 0.6):
        self.cycle_approach = cycle_approach
        self.train_epochs = train_epochs
        self.distill_percent = distill_percent
        self.distill_epoch = int(train_epochs * distill_percent)
        
        
        # distill 10 epochs / gap / distill 10 epochs
        if self.cycle_approach == 'ten_to_ten':
            self.n_gap = self.distill_epoch // 10
            self.gap = int(self.train_epochs * (1 - self.distill_percent) / self.n_gap)
            self.ten_to_ten_list = []
            for point in range(0,200, 10+ self.gap):
                self.ten_to_ten_list.extend(range(point, point+10))

    def _get_distill_use(self, epoch, total_loss = 0, train_loss = 0, kd_loss = 0):
        if self.cycle_approach == 'all':
            return True, False
        
        elif self.cycle_approach == 'none':
            return False, False
        
        elif self.cycle_approach == 'first':
            if epoch < self.distill_epoch :
                return True, False
            else:
                return False, False
    
        elif self.cycle_approach == 'mid':
            if epoch >= self.train_epochs // 2 - self.distill_epoch // 2 and epoch < self.train_epochs // 2 + self.distill_epoch // 2:
                return True, False
            else:
                return False, False

        elif self.cycle_approach == 'end':
            if epoch >= self.train_epochs - self.distill_epoch:
                return True, False
            else:
                return False, False
        
        elif self.cycle_approach == 'first_end':
            if (epoch < self.distill_epoch // 2) or (epoch>= self.train_epochs - self.distill_epoch//2):
                return True, False
            else:
                return False, False
        
        elif self.cycle_approach == 'ten_to_ten':
            if epoch in self.ten_to_ten_list:
                return True, False
            else:
                return False, False
        
        elif self.cycle_approach == 'cycle':
            if self.distill_percent < 0.5:
                if self.distill_percent == 0.2:
                    if epoch % 5 == 0:
                        return True, False
                    else:
                        return False, False
                elif self.distill_percent == 0.4:
                    if epoch % 5 in [1,3]:
                        return True, False
                    else:
                        return False, False
            else:
                if self.distill_percent == 0.8:
                    if epoch %  5 != 0:
                        return True, False
                    else:
                        return False, False
                elif self.distill_percent == 0.6:
                    if epoch % 5 in [0,2,4]:
                        return True, False
                    else:
                        return False, False
        
        elif self.cycle_approach == 'custom':
            return self._make_distill_cycle()[epoch], ~self._make_distill_cycle()[epoch]
        
        elif self.cycle_approach == 'adaptive':
            return self._get_adaptive_cycle(total_loss, train_loss, kd_loss), ~self._get_adaptive_cycle(total_loss, train_loss, kd_loss)
        
        ### IKR (DKD가 부여되지 않음)
        elif self.cycle_approach is None:
            return False, True  ### 수정 필요 (for IKR)

    def _get_adaptive_cycle(self, total_loss, train_loss, kd_loss):
        return True, False

    def _make_distill_cycle(self, repeat_num=5, repeat_num_decrease=-1, repeat_period=80, repeat_delay=5, repeat_period_change=-10):
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
        
        distill_cycle = [False] * self.train_epochs
        repeat_num = repeat_num
        for i in range(repeat_delay, self.train_epochs, repeat_period):
            for j in range(repeat_num):
                distill_cycle[i + j] = True
            repeat_num += repeat_num_decrease
            repeat_period += repeat_period_change
        return distill_cycle
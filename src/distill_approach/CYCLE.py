class CYCLE():
    def __init__(self, cycle_approach, train_epochs = 200):
        self.cycle_approach = cycle_approach
        self.train_epochs = train_epochs

    def _get_distill_use(self, epoch, total_loss = 0, train_loss = 0, kd_loss = 0):
        if self.cycle_approach == 'all':
            return True, False
        
        elif self.cycle_approach == 'none':
            return False, False
        
        elif self.cycle_approach == 'first_20':
            return True,False if epoch < 20 else False, True
    
        elif self.cycle_approach == 'mid_20':
            return True, False if epoch >= 90 and epoch < 110 else False, True
        elif self.cycle_approach == 'end_20':

            return True, False if epoch >= 180 else False, True
        
        elif self.cycle_approach == 'every_10':
            return True, False if epoch % 10 != 0 else False, True
        
        elif self.cycle_approach == 'custom':
            return self._make_distill_cycle()[epoch], ~self._make_distill_cycle()[epoch]
        
        elif self.cycle_approach == 'adaptive':
            return self._get_adaptive_cycle(total_loss, train_loss, kd_loss), ~self._get_adaptive_cycle(total_loss, train_loss, kd_loss)
        

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
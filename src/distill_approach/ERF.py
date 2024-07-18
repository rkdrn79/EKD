class ERF():
    def __init__(self, erf_approach, train_epochs, m = 1):
        self.erf_approach = erf_approach
        self.train_epochs = train_epochs
        self.m = m
        self.erf_distill_cycle = self._get_erf_distill_cycle()
        self.weight_memory = [None] * self.train_epochs

    def _get_distill_use(self, epoch):
        return self.erf_distill_cycle[epoch]
    
    def _get_distill_weight(self, epoch, total_loss, kd_loss):
        if self.erf_approach == 'adaptive_all' or self.erf_approach == 'adaptive':
            return self._make_addaptive_distill_weight(total_loss, kd_loss)
        else:
            return self.weight_memory[epoch]
    
    def _get_erf_distill_cycle(self):
        if self.erf_approach == 'adaptive_all':
            return [True] * self.train_epochs
        
        elif self.erf_approach == 'every_none':
            return [False] * self.train_epochs
        
        elif self.erf_approach == 'first_20':
            return [True] * 20 + [False] * (self.train_epochs - 20)
        
        elif self.erf_approach == 'mid_20':
            return [False] * (self.train_epochs//2 - 10) + [True] * 20 + [False] * (self.train_epochs//2 - 10)
        
        elif self.erf_approach == 'end_20':
            return [False] * (self.train_epochs - 20) + [True] * 20
        
        elif self.erf_approach == 'every_10':
            return [True if i % 10 == 0 else False for i in range(self.train_epochs)]
        
        else:
            return self._make_distill_cycle()

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

    def _make_addaptive_distill_weight(self, total_loss, kd_loss):
        return (kd_loss / total_loss) * self.m
class ERF():
    def __init__(self, erf_approach, m = 2):
        self.erf_approach = erf_approach
        self.m = m
    
    def _get_distill_weight(self, epoch = 0, train_loss = 1, kd_loss = 1):
        if self.erf_approach == 'adaptive':
            return self._make_addaptive_distill_weight(train_loss, kd_loss) * self.m
        else:
            return 1 * self.m

    def _make_addaptive_distill_weight(self, total_loss, kd_loss):
        return (kd_loss / total_loss)
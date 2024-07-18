class RGR():
    def __init__(self, rgr_approach, train_epochs, erf_distill_cycle, ema_alpha = 0.9):
        self.rgr_approach = rgr_approach
        self.train_epochs = train_epochs
        self.ema_alpha = ema_alpha
        self.weight_memory = [1] * self.train_epochs
        self.rgr_distill_cycle = self._get_rgr_distill_cycle(erf_distill_cycle)
        self.ema_gradidents = None
        self.t = None

    def _get_distill_use(self, epoch):
        return self.rgr_distill_cycle[epoch]
    
    def _get_rgr_distill_cycle(self, erf_distill_cycle):
        """
        erf distill cycle -> True => rgr distiil cycle -> False
        erf distill cycle -> False => rgr distill cycle -> True 
        """
        if self.rgr_approach == 'none':
            return [False] * self.train_epochs
        else:
            return [not x for x in erf_distill_cycle]
        
    def _get_ema(self, epoch, model):
        if self.ema_gradidents is not None:
            for name, param in model.named_parameters():
                if name in self.ema_gradidents.keys():
                    if self.ema_gradidents[name] is not None:
                        param.grad =  self.ema_gradidents[name] * self._get_distill_weight(epoch)
                    else:
                        param.grad = None
        return model
        
    def _save_ema(self, loss, optimizer, model):
        current_gradients = dict()
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)

        if model.training:
            for name, param in model.named_parameters(): # ema_gradidents is None -> save current gradients
                if param.grad is not None:
                    current_gradients[name] = param.grad.clone().detach()

            if self.ema_gradidents: # ema_gradidents is not None -> update ema_gradidents
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.ema_gradidents[name] = self.ema_alpha * self.ema_gradidents[name] + (1 - self.ema_alpha) * current_gradients[name]

            else: # ema_gradidents is None
                self.ema_gradidents = current_gradients

    def _reset_eam(self, t):
        if t != self.t:
            self.ema_gradidents = None
            self.t = t
        
    def _get_distill_weight(self, epoch):
        return self.weight_memory[epoch]
    
    
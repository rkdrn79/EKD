class RGR():
    def __init__(self, rgr_approach, ema_alpha = 0.9, m = 1):
        self.rgr_approach = rgr_approach
        self.ema_alpha = ema_alpha
        self.m = m
        self.ema_gradidents = None
        self.t = 0

    
    def _get_ema(self, epoch, model):
        if self.ema_gradidents is not None:
            for name, param in model.named_parameters():
                if name in self.ema_gradidents.keys():
                    if self.ema_gradidents[name] is not None:
                        param.grad =  self.ema_gradidents[name] * self._get_distill_weight(epoch) * self.m
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
        
    def _get_distill_weight(self, total_loss, kd_loss):
        if self.rgr_approach == 'adaptive':
            return self._make_addaptive_distill_weight(total_loss, kd_loss)
        else:
            return 1

    def _make_addaptive_distill_weight(self, total_loss, kd_loss):
        return (kd_loss / total_loss) * self.m
    
    
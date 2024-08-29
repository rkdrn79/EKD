import math

class ERF():
    def __init__(self, erf_approach, m = 2):
        self.erf_approach = erf_approach
        self.m = m
    
    def _get_distill_weight(self, epoch = 0, train_loss = 1, kd_loss = 1, distill_percent = 0.2, n_epochs=200, cycle_approach = 'mid'):
        # Code needs to be more concise and looking good
        # Need to go through chat gpt after implementing all the functions including cycle approach
        if self.erf_approach == 'convexed_increase':
            if cycle_approach == 'cycle':
                return self.m * math.sin(math.pi * (epoch) / (2 * n_epochs)+ 1.5 * math.pi) + self.m
            elif cycle_approach == 'first_end':
                if epoch > n_epochs/2:
                    return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent/2)) / (n_epochs * distill_percent)+ 1.5 * math.pi)+ self.m
                else:
                    return self.m * math.sin(math.pi * (epoch) / (n_epochs * distill_percent)+ 1.5 * math.pi) + self.m
            elif cycle_approach == 'first':
                # epoch = x
                return self.m * math.sin(math.pi * (epoch) / (2 * n_epochs * distill_percent)+ 1.5 * math.pi) + self.m
            elif cycle_approach == 'mid':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent) / 2) / (2 * n_epochs * distill_percent) + 1.5 * math.pi) + self.m
            elif cycle_approach == 'end':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent)) / (2 * n_epochs * distill_percent)+ 1.5 * math.pi)+ self.m
            elif cycle_approach == 'ten_to_ten':
                n_gap = int (n_epochs * distill_percent) // 10
                gap = int (n_epochs * (1-distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return self.m * math.sin(math.pi * (epoch - starting_point) / (2 * (10)) + 1.5 * math.pi) + self.m
            else:
                return self.m
            
        elif self.erf_approach == 'convexed_decrease':
            if cycle_approach == 'cycle':
                return self.m * math.sin(math.pi * (epoch) / (2 * n_epochs)+ math.pi) + self.m
            elif cycle_approach == 'first_end':
                if epoch > n_epochs/2:
                    return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent/2)) / (n_epochs * distill_percent)+ math.pi)+ self.m
                else:
                    return self.m * math.sin(math.pi * (epoch) / (n_epochs * distill_percent)+ math.pi) + self.m
            elif cycle_approach == 'first':
                # epoch = x
                return self.m * math.sin(math.pi * (epoch) / (2 * n_epochs * distill_percent)+ math.pi) + self.m
            elif cycle_approach == 'mid':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent) / 2) / (2 * n_epochs * distill_percent)+ math.pi) + self.m
            elif cycle_approach == 'end':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent)) / (2 * n_epochs * distill_percent)+ math.pi)+ self.m
            elif cycle_approach == 'ten_to_ten':
                n_gap = int (n_epochs * distill_percent) // 10
                gap = int (n_epochs * (1-distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return self.m * math.sin(math.pi * (epoch-starting_point) / (2 * (10)) + math.pi) + self.m
            else:
                return self.m
            
        elif self.erf_approach == 'concaved_increase':
            if cycle_approach == 'cycle':
                return self.m * math.sin(math.pi * epoch / (2 * n_epochs))
            elif cycle_approach == 'first_end':
                if epoch > n_epochs/2:
                    return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent/2)) / (n_epochs * distill_percent))
                else:
                    return self.m * math.sin(math.pi * epoch / (n_epochs * distill_percent))
            elif cycle_approach == 'first':
                # epoch = x
                return self.m * math.sin(math.pi * epoch / (2 * n_epochs * distill_percent))
            elif cycle_approach == 'mid':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent) / 2) / (2 * n_epochs * distill_percent)) 
            elif cycle_approach == 'end':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent)) / (2 * n_epochs * distill_percent))
            elif cycle_approach == 'ten_to_ten':
                n_gap = int (n_epochs * distill_percent) // 10
                gap = int (n_epochs * (1-distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return math.sin(math.pi * (epoch - starting_point) / (2 * (10)))
            else:
                return 1 * self.m
            
        elif self.erf_approach == 'concaved_decrease':
            if cycle_approach == 'cycle':
                return self.m * math.sin(math.pi * (epoch) / (2 * n_epochs)+ 0.5 * math.pi)
            elif cycle_approach == 'first_end':
                if epoch > n_epochs/2:
                    return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent/2)) / (n_epochs * distill_percent) + math.pi/2)
                else:
                    return self.m * math.sin(math.pi * (epoch) / (n_epochs * distill_percent) + 0.5 * math.pi)
            elif cycle_approach == 'first':
                # epoch = x
                return self.m * math.sin(math.pi * (epoch) / (2 * n_epochs * distill_percent)+ 0.5 * math.pi)
            elif cycle_approach == 'mid':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent) / 2) / (2 * n_epochs * distill_percent)+ math.pi/2) 
            elif cycle_approach == 'end':
                return self.m * math.sin(math.pi * (epoch - n_epochs * (1 - distill_percent)) / (2 * n_epochs * distill_percent)+ math.pi/2)
            elif cycle_approach == 'ten_to_ten':
                n_gap = int (n_epochs * distill_percent) // 10
                gap = int (n_epochs * (1-distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return math.sin(math.pi * (epoch - starting_point) / (2 * (10)) + 0.5 * math.pi)
            else:
                return self.m
            
        elif self.erf_approach == 'linear_increase':
            if cycle_approach == 'cycle':
                return self.m /n_epochs * epoch
            elif cycle_approach == 'first_end':
                if epoch > n_epochs/2:
                    return self.m / (n_epochs * distill_percent/2) * (epoch - n_epochs * (1 - distill_percent))
                else:
                    return self.m / (n_epochs * distill_percent/2) * epoch
            elif cycle_approach == 'first':
                # epoch = x
                return self.m / (n_epochs * distill_percent) * epoch
            elif cycle_approach == 'mid':
                return self.m / (n_epochs * distill_percent) * (epoch - n_epochs * (1 - distill_percent) / 2)
            elif cycle_approach == 'end':
                return self.m / (n_epochs * distill_percent) * (epoch - n_epochs * (1 - distill_percent))
            elif cycle_approach == 'ten_to_ten':
                n_gap = int (n_epochs * distill_percent) // 10
                gap = int (n_epochs * (1-distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return self.m / 10 * (epoch - starting_point)
            else:
                return self.m
        elif self.erf_approach == 'linear_decrease':
            if cycle_approach == 'cycle':
                return -1 * self.m / (n_epochs) * epoch + self.m
            elif cycle_approach == 'first_end':
                if epoch > n_epochs/2:
                    return -1 * self.m / (n_epochs * distill_percent/2) * epoch + self.m
                else:
                    return -1 * self.m / (n_epochs * distill_percent/2) * (epoch - n_epochs * (1 - distill_percent/2)) + self.m
            elif cycle_approach == 'first':
                # epoch = x
                return -1 * self.m / (n_epochs * distill_percent) * epoch + self.m
            elif cycle_approach == 'mid':
                return -1 * self.m / (n_epochs * distill_percent) * (epoch - n_epochs * (1 - distill_percent) / 2) + self.m
            elif cycle_approach == 'end':
                return -1 * self.m / (n_epochs * distill_percent) * (epoch - n_epochs * (1 - distill_percent)) + self.m
            elif cycle_approach == 'ten_to_ten':
                n_gap = int (n_epochs * distill_percent) // 10
                gap = int (n_epochs * (1-distill_percent) / n_gap)
                starting_point = max([point for point in range(0,200, 10+gap) if point <= epoch])
                
                return -1 * self.m / 10 * (epoch - starting_point) + self.m
            else:
                return self.m
        else:
            return self.m


    def _make_addaptive_distill_weight(self, total_loss, kd_loss):
        return (kd_loss / total_loss)
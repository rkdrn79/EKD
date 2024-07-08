import numpy as np
import matplotlib.pyplot as plt

class ERF_Distillation():
    def __init__(self, distill_approach, train_epochs, memory_loss_use):
        self.distill_approach = distill_approach
        self.memory_loss_use = memory_loss_use
        self.train_epochs = train_epochs
        self.total_arr = self.make_total_arr()
        self.total_arr2 = self.smooth_decay(self.total_arr)
    
    def loss(self, epoch):
        if self.memory_loss_use == "none":
            if self.total_arr[epoch] == 0:
                return False, 0, 0  
            else:
                return True, self.total_arr[epoch], 0
        else:
            if self.total_arr[epoch] == 0:
                return False, 0, self.total_arr2[epoch]
            else:
                return True, self.total_arr[epoch], 0

    def make_total_arr(self):
        if self.distill_approach == 'every':
            return [1] * self.train_epochs
        
        elif self.distill_approach == 'every_none':
            return [0] * self.train_epochs
        
        elif self.distill_approach == 'first_20':
            return [1] * 20 + [0] * (self.train_epochs - 20)
        
        elif self.distill_approach == 'mid_20':
            return [0] * (self.train_epochs//2 - 10) + [1] * 20 + [0] * (self.train_epochs//2 - 10)
        
        elif self.distill_approach == 'end_20':
            return [0] * (self.train_epochs - 20) + [1] * 20
        
        elif self.distill_approach == 'every_20':
            return [1 if i % 20 == 0 else 0 for i in range(self.train_epochs)]
        
        elif self.distill_approach == 'first_10_end_10':
            result = [0] * self.train_epochs
            result[:10] = [1] * 10  # 처음 10개
            result[-10:] = [1] * 10  # 마지막 10개
            return result
        
        elif self.distill_approach == 'custom1':
            arr = np.arange(0,3,0.4)
            arr_exp = np.exp(arr)
            arr_transformed =  - (arr_exp * 2 / max(arr_exp) - 2)
            arr = np.concatenate((np.zeros(5), arr_transformed))
            arr2 = np.zeros(80)
            total_arr = self.kd_strategy_maker(arr, arr2, total_epoch=200, plot=False, strategy_num=1)
            return total_arr
        
        elif self.distill_approach == 'custom2':
            arr = np.arange(0,3,0.4)
            arr_exp = np.exp(arr)
            arr_transformed =  - (arr_exp * 2 / max(arr_exp) - 2)
            arr = np.concatenate((np.zeros(5), arr_transformed))
            arr2 = np.zeros(15)
            total_arr = self.kd_strategy_maker2(arr, arr2, total_epoch=200,arr_decrease_weight = 0.9, arr_decrease_length = 1 , arr2_increase_length = 19, plot=False, strategy_num=2)
            return total_arr
        
        elif self.distill_approach == 'custom3':
            arr = np.arange(0,3,0.4)
            arr = np.concatenate((np.zeros(5), 1.5*np.log(arr + 1)))
            arr2 = np.zeros(80)
            total_arr = self.kd_strategy_maker(arr, arr2, total_epoch=200, plot=True, strategy_num=1)
            return total_arr
        
        elif self.distill_approach == 'custom_final':
            arr = np.arange(0, 3, 0.4)
            arr = np.exp(arr)
            arr =  - (arr * 2 / max(arr) - 2)
            total_arr = self.kd_strategy_maker_final(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.)
            return total_arr
        
        elif self.distill_approach == 'custom_final2':
            # strategy 1
            arr = np.arange(0,3, 0.4)
            arr = np.concatenate((np.zeros(5), 1.5*np.log(arr + 1)))
            total_arr = self.kd_strategy_maker_final(arr, repeat_period=80, repeat_delay=5, repeat_period_change=-10, repeat_weight_change=1.)
            return total_arr

    def kd_strategy_maker(self, arr, arr2, total_epoch=200, plot=False , strategy_num=None):
        repeat_num=total_epoch//(len(arr)+len(arr2))
        
        # arr와 arr2의 길이 합으로 total_epoch이 나머지 없이 나뉘는 경우
        if (total_epoch%(len(arr)+len(arr2)))==0:   
            # print('Case1')
            total_arr=np.concatenate([arr,arr2]*repeat_num)
        # arr와 arr2의 길이 합으로 total_epoch이 나머지 있게 나뉘는 경우
        else:
            # 나머지가 arr의 길이보다 긴 경우
            if len(arr)< ( total_epoch%(len(arr)+len(arr2)) ):
                # print('Case2')
                total_arr=np.concatenate([arr,arr2]*repeat_num+[arr]+[arr2[:(total_epoch%(len(arr)+len(arr2)) - len(arr))]])
            # 나머지가 arr의 길이보다 짧은 경우
            else:
                # print('Case3')
                total_arr=np.concatenate([arr,arr2]*repeat_num+[arr[:total_epoch%(len(arr)+len(arr2))]])
                
        # 플로팅 여부
        if plot:
            plt.figure(figsize=(10,2))
            if strategy_num ==None:
                plt.title('KD Epoch Strategy')
            else:
                plt.title(f'KD Epoch Strategy {strategy_num}')
            plt.plot(total_arr)
            plt.xlabel('epoch')
            plt.ylabel('weight')
            plt.show()
        return total_arr
    
    def kd_strategy_maker2(self ,arr, arr2, total_epoch=200, arr_decrease_weight = 0.9, arr_decrease_length = 1  , arr2_increase_length =10, plot=True, strategy_num=None):
        total_length = len(arr) + len(arr2)
        repeat_num = total_epoch // total_length
        remaining_epochs = total_epoch % total_length
        
        total_arr = []

        # Repeat the sequence while modifying arr and arr2
        for i in range(repeat_num):
            arr = arr * arr_decrease_weight # Decrease arr's values
            if len(arr) > 1:
                arr = arr[:len(arr)-arr_decrease_length]  # Decrease arr's length
            arr2 = np.zeros(len(arr2) + arr2_increase_length)  # Increase arr2's length
            total_arr.extend(arr)
            total_arr.extend(arr2)
        
        # Handle remaining epochs
        if remaining_epochs > len(arr):
            total_arr.extend(arr)
            remaining_epochs -= len(arr)
            total_arr.extend(arr2[:remaining_epochs])
        else:
            total_arr.extend(arr[:remaining_epochs])
        
        total_arr = np.array(total_arr[:total_epoch])
        
        # Plotting
        if plot:
            plt.figure(figsize=(10, 2))
            if strategy_num is None:
                plt.title('KD Epoch Strategy')
            else:
                plt.title(f'KD Epoch Strategy {strategy_num}')
            plt.plot(total_arr)
            plt.xlabel('epoch')
            plt.ylabel('weight')
            plt.show()

        return total_arr

    def kd_strategy_maker_final(self, repeat_arr, repeat_period, repeat_delay=0, repeat_period_change=0, repeat_weight_change=1., threshold=0, plot=True, strategy_num=None, total_epoch=200, ):
        """Codes for kd strategy
        Args:
            repeat_arr : 복습의 형태를 정의하는 array
            repeat_period : 복습 간 주기
            repeat_delay : 첫 에폭에 시작하지 않고 약간 delay한 복습
            repeat_period_change : 복습 간 주기의 변화량 (덧셈)
            total_epoch : 총 학습 에폭
            repeat_weight_change : 복습 정도의 증감폭 (곱셈)
            threshold : 너무 작은 kd weight는 0으로 줄이는 것, 효율성 증가
        """
        total_arr = np.array([0.] * total_epoch)

        no_weight, give_weight = 0, len(repeat_arr)

        for e in range(total_epoch):
            if e < repeat_delay:
                continue

            if no_weight:  # (N, 0)
                total_arr[e] = 0
                no_weight += -1
                if no_weight == 0:  # (1, 0) -> (0, 0)
                    give_weight = len(repeat_arr)
                    repeat_period += repeat_period_change

            elif give_weight:  # (0, N)
                total_arr[e] = repeat_arr[-give_weight]
                give_weight += -1
                if give_weight == 0:  # (0, 1) -> (0, 0)
                    repeat_arr *= repeat_weight_change
                    no_weight = repeat_period

        # Plotting
        if plot:
            plt.figure(figsize=(10, 2))
            if strategy_num is None:
                plt.title('KD Epoch Strategy')
            else:
                plt.title(f'KD Epoch Strategy {strategy_num}')
            plt.plot(total_arr)
            plt.xlabel('epoch')
            plt.ylabel('weight')
            plt.show()

        return total_arr

    
    def smooth_decay(self, total_arr):
        smooth_arr = total_arr.copy()  # 원본 배열 복사
    
        # 0이 아닌 값을 가지는 인덱스 찾기
        nonzero_indices = np.where(total_arr != 0)[0]
        
        for i in range(len(nonzero_indices) - 1):
            start_idx = nonzero_indices[i]
            end_idx = nonzero_indices[i + 1]
            
            # 시작 값과 끝 값
            start_val = total_arr[start_idx]
            end_val = total_arr[end_idx]
            
            # 지수 감소 함수 생성
            decay_range = np.arange(start_idx, end_idx)
            decay_func = start_val * np.exp(-(decay_range - start_idx) / (end_idx - start_idx))
            
            # 해당 구간에 지수 감소 함수 적용
            smooth_arr[start_idx:end_idx] = decay_func
        
        # 마지막 0이 아닌 값 이후의 값들에 대해 지수 감소 함수 적용
        if len(nonzero_indices) > 0:
            last_idx = nonzero_indices[-1]
            start_val = total_arr[last_idx]
            decay_range = np.arange(last_idx, len(total_arr))
            decay_func = start_val * np.exp(-(decay_range - last_idx) / len(total_arr))
            smooth_arr[last_idx:] = decay_func
        
        return smooth_arr
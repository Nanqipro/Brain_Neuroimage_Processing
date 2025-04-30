from math import cos, pi

class CosineLrUpdater():
    """
    余弦学习率调度器
    
    实现学习率的余弦退火策略，支持预热和重启功能
    
    参数
    ----------
    optimizer : torch.optim.Optimizer
        优化器对象
    periods : list[int]
        各阶段的周期长度
    by_epoch : bool, 可选
        是否按照epoch更新学习率，默认为True
    warmup : str, 可选
        预热类型，可选值：'constant'、'linear'或'exp'，默认为None
    warmup_iters : int, 可选
        预热迭代次数，默认为0
    warmup_ratio : float, 可选
        预热开始时的学习率比例，默认为0.1
    warmup_by_epoch : bool, 可选
        预热是否按照epoch计算，默认为False
    restart_weights : list[float], 可选
        重启权重列表，默认为[1]
    min_lr : list[float], 可选
        最小学习率列表，与min_lr_ratio二选一
    min_lr_ratio : float, 可选
        最小学习率比例，与min_lr二选一
    """
    
    def __init__(self,
                 optimizer,
                 periods,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None):
        
        assert (min_lr is None) ^ (min_lr_ratio is None)
       
        # 验证warmup参数
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        self.optimizer = optimizer
        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'

        # 设置预热参数
        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # 所有参数组的初始学习率
        self.regular_lr = []  # 预期的学习率（如果没有预热）

        # 计算累积周期
        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def _set_lr(self, lr_groups):
        """
        设置学习率
        
        参数
        ----------
        lr_groups : list[float]
            各参数组的学习率列表
        """
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, epoch, iter, base_lr):
        """
        获取当前学习率
        
        参数
        ----------
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
        base_lr : list[float]
            基础学习率列表
            
        返回值
        ----------
        list[float]
            计算后的学习率列表
        """
        if self.by_epoch:
            progress = epoch
        else:
            progress = iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr[idx], alpha, current_weight)


    def get_regular_lr(self, epoch, iter):
        """
        获取常规学习率（不含预热）
        
        参数
        ----------
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
            
        返回值
        ----------
        list[float]
            常规学习率列表
        """
        return [self.get_lr(epoch, iter, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        """
        获取预热阶段的学习率
        
        参数
        ----------
        cur_iters : int
            当前迭代次数
            
        返回值
        ----------
        list[float]
            预热阶段的学习率列表
        """
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self):
        """
        训练开始前的初始化
        
        获取优化器中所有参数组的初始学习率
        """
        # 当从检查点恢复时，如果'initial_lr'未保存，则根据优化器参数设置
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

    def before_train_epoch(self, len_train_data, epoch, iter):
        """
        每个训练epoch开始前调用
        
        参数
        ----------
        len_train_data : int
            训练数据长度
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
        """
        if not self.by_epoch:
            return
        if self.warmup_by_epoch:
            epoch_len = len_train_data
            self.warmup_iters = self.warmup_epochs * epoch_len

        self.regular_lr = self.get_regular_lr(epoch, iter)
        self._set_lr(self.regular_lr)

    def before_train_iter(self, epoch, iter):
        """
        每次训练迭代前调用
        
        参数
        ----------
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
        """
        cur_iter = iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(epoch, iter)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(warmup_lr)

def get_position_from_periods(iteration, cumulative_periods):
    """
    从周期列表中获取位置
    
    返回周期列表中右侧最近数字的索引
    例如，当cumulative_periods = [100, 200, 300, 400]时：
    如果iteration == 50，返回0；
    如果iteration == 210，返回2；
    如果iteration == 300，返回3。
    
    参数
    ----------
    iteration : int
        当前迭代次数
    cumulative_periods : list[int]
        累积周期列表
        
    返回值
    ----------
    int
        周期列表中右侧最近数字的位置
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


def annealing_cos(start, end, factor, weight=1):
    """
    计算余弦退火学习率
    
    当百分比从0.0变化到1.0时，从`weight * start + (1 - weight) * end`衰减到`end`
    
    参数
    ----------
    start : float
        余弦退火的起始学习率
    end : float
        余弦退火的结束学习率
    factor : float
        计算当前百分比时`pi`的系数，范围从0.0到1.0
    weight : float, 可选
        计算实际起始学习率时`start`和`end`的组合因子，默认为1
        
    返回值
    ----------
    float
        计算后的学习率
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out

"""
学习率更新器模块

该模块实现了余弦学习率调度器，支持学习率预热和周期性重启策略。
可以按照epoch或iteration更新学习率，提供多种预热方式选择。

作者: SCN研究小组
日期: 2023
"""

from math import cos, pi

class CosineLrUpdater():
    """
    余弦学习率调度器
    
    实现了带预热和周期性重启的余弦退火学习率调度策略，
    可以在训练过程中动态调整学习率，提高模型性能。
    
    参数
    ----------
    optimizer : torch.optim.Optimizer
        优化器实例，用于更新学习率
    periods : list[int]
        每个周期的长度列表（按epoch或iteration计算）
    by_epoch : bool, 可选
        是否按epoch更新学习率，默认为True
    warmup : str, 可选
        预热类型，可选值：None（不使用预热）、'constant'、'linear'或'exp'，默认为None
    warmup_iters : int, 可选
        预热迭代次数，默认为0
    warmup_ratio : float, 可选
        预热起始学习率比例，为初始学习率的倍数，默认为0.1
    warmup_by_epoch : bool, 可选
        预热是否按epoch计算，默认为False
    restart_weights : list[float], 可选
        每个周期重启时的权重列表，默认为[1]
    min_lr : list[float], 可选
        最小学习率列表，与min_lr_ratio互斥
    min_lr_ratio : float, 可选
        最小学习率比例，与min_lr互斥
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
        
        # 确保min_lr和min_lr_ratio只指定一个
        assert (min_lr is None) ^ (min_lr_ratio is None)
       
        # 验证预热类型参数
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
                
        # 初始化参数
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
        # 确保周期和重启权重长度一致
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'

        # 如果按epoch计算预热，转换预热参数
        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        # 初始化学习率列表
        self.base_lr = []  # 所有参数组的初始学习率
        self.regular_lr = []  # 预期学习率（无预热时）

        # 计算累积周期长度
        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def _set_lr(self, lr_groups):
        """
        设置优化器的学习率
        
        参数
        ----------
        lr_groups : list[float]
            每个参数组的学习率列表
        """
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, epoch, iter, base_lr):
        """
        计算当前的学习率
        
        基于当前的训练进度和基础学习率，使用余弦退火策略计算学习率。
        
        参数
        ----------
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
        base_lr : list[float]
            基础学习率列表
            
        返回
        -------
        float
            计算得到的当前学习率
        """
        # 根据by_epoch决定使用epoch还是iter作为进度指标
        if self.by_epoch:
            progress = epoch
        else:
            progress = iter

        # 确定目标最小学习率
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        # 确定当前所在周期索引
        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        # 计算周期内的相对进度（0到1之间）
        alpha = min((progress - nearest_restart) / current_periods, 1)
        # 使用余弦退火公式计算学习率
        return annealing_cos(base_lr, target_lr[idx], alpha, current_weight)


    def get_regular_lr(self, epoch, iter):
        """
        获取所有参数组的常规学习率（无预热时）
        
        参数
        ----------
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
            
        返回
        -------
        list[float]
            所有参数组的学习率列表
        """
        return [self.get_lr(epoch, iter, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        """
        获取预热阶段的学习率
        
        根据预热类型和当前迭代次数计算预热阶段的学习率。
        
        参数
        ----------
        cur_iters : int
            当前迭代次数
            
        返回
        -------
        list[float]
            预热阶段的学习率列表
        """
        if self.warmup == 'constant':
            # 常数预热：使用固定的warmup_ratio * 常规学习率
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            # 线性预热：从warmup_ratio * 常规学习率线性增加到常规学习率
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            # 指数预热：从warmup_ratio * 常规学习率指数增加到常规学习率
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self):
        """
        在训练开始前调用，初始化基础学习率
        """
        # 注意：从断点恢复时，如果'initial_lr'未保存，将根据优化器参数设置
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in self.optimizer.param_groups
        ]

    def before_train_epoch(self, len_train_data, epoch, iter):
        """
        在每个训练epoch开始前调用，更新学习率
        
        参数
        ----------
        len_train_data : int
            训练数据的长度（批次数）
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
        """
        if not self.by_epoch:
            return
        # 如果按epoch计算预热，计算预热迭代次数
        if self.warmup_by_epoch:
            epoch_len = len_train_data
            self.warmup_iters = self.warmup_epochs * epoch_len

        # 计算常规学习率并设置
        self.regular_lr = self.get_regular_lr(epoch, iter)
        self._set_lr(self.regular_lr)

    def before_train_iter(self, epoch, iter):
        """
        在每次训练迭代前调用，更新学习率
        
        参数
        ----------
        epoch : int
            当前epoch
        iter : int
            当前迭代次数
        """
        cur_iter = iter
        if not self.by_epoch:
            # 如果不按epoch更新，则每次迭代都更新学习率
            self.regular_lr = self.get_regular_lr(epoch, iter)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                # 非预热阶段，使用常规学习率
                self._set_lr(self.regular_lr)
            else:
                # 预热阶段，使用预热学习率
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(warmup_lr)
        elif self.by_epoch:
            # 如果按epoch更新，只在预热阶段更新学习率
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                # 预热结束，切换到常规学习率
                self._set_lr(self.regular_lr)
            else:
                # 预热阶段，使用预热学习率
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(warmup_lr)

def get_position_from_periods(iteration, cumulative_periods):
    """
    从累积周期列表中获取位置索引
    
    返回周期列表中右侧最接近的数字的索引。
    例如，当cumulative_periods = [100, 200, 300, 400]时：
    - 如果iteration == 50，返回0
    - 如果iteration == 210，返回2
    - 如果iteration == 300，返回3
    
    参数
    ----------
    iteration : int
        当前迭代次数或epoch
    cumulative_periods : list[int]
        累积周期列表
        
    返回
    -------
    int
        周期列表中右侧最接近数字的索引
        
    异常
    -------
    ValueError
        当当前迭代次数超过累积周期列表的最大值时抛出
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


def annealing_cos(start, end, factor, weight=1):
    """
    计算余弦退火学习率
    
    从`weight * start + (1 - weight) * end`到`end`进行余弦退火，
    随着factor从0.0增加到1.0。
    
    参数
    ----------
    start : float
        余弦退火的起始学习率
    end : float
        余弦退火的结束学习率
    factor : float
        计算当前百分比时的pi系数，范围从0.0到1.0
    weight : float, 可选
        计算实际起始学习率时start和end的组合因子，默认为1
        
    返回
    -------
    float
        计算得到的当前学习率
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out

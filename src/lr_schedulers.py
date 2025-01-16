import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """
    https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): First cycle step size.
        T_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        eta_min(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: float = 1.0,
        max_lr: float = 0.001,
        eta_min: float = 0.0001,
        warmup_steps: int = 5,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < T_0

        self.T_0 = T_0  # first cycle step size
        self.T_mult = T_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.eta_min = eta_min  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = T_0  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmRestartsWithWarmup, self).__init__(optimizer, last_epoch)

        # set learning rate eta_min
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.T_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1.0:
                    self.step_in_cycle = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.T_0 * (self.T_mult - 1)
                                + 1
                            ),
                            self.T_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.T_0
                        * (self.T_mult**n - 1)
                        / (self.T_mult - 1)
                    )
                    self.cur_cycle_steps = self.T_0 * self.T_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.T_0
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class WarmupPiecewiseConstantScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        eta_min,
        max_lr,
        milestones_lrs,
        last_epoch=-1,
        verbose=True,
    ):
        """
        Custom learning rate scheduler with linear warmup and piecewise constant schedule.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for linear warmup.
            eta_min (float): Starting learning rate for warmup.
            max_lr (float): Learning rate after warmup.
            milestones_lrs (list of tuples): List where each tuple is (milestone_step, lr_value).
                After warmup, the learning rate changes to lr_value at milestone_step.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.max_lr = max_lr
        self.milestones_lrs = milestones_lrs
        super(WarmupPiecewiseConstantScheduler, self).__init__(
            optimizer, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            warmup_progress = self.last_epoch / self.warmup_steps
            lr = self.eta_min + warmup_progress * (self.max_lr - self.eta_min)
            return [lr for _ in self.base_lrs]
        else:
            # Piecewise constant learning rate after warmup
            lr = self.max_lr
            for milestone_step, lr_value in self.milestones_lrs[::-1]:
                if self.last_epoch >= milestone_step:
                    lr = lr_value
                    break
            return [lr for _ in self.base_lrs]
        
class ExponentialWithMinLR(torch.optim.lr_scheduler._LRScheduler):
  """Adopted from: https://github.com/Liu-Feng-deeplearning/CoverHunter/blob/main/src/scheduler.py
  Decays the learning rate of each parameter group by _gamma every epoch.
  When last_epoch=-1, sets initial lr as lr.

  Args:
      optimizer (Optimizer): Wrapped optimizer.
      gamma (float): Multiplicative factor of learning rate decay.
      eta_min(float): min lr.
      last_epoch (int): The index of last epoch. Default: -1.

  """
  def __init__(self, optimizer, gamma, eta_min, last_epoch=-1, warmup_steps=None):
    self.gamma = gamma
    self.eta_min = eta_min
    self.warmup_steps = warmup_steps
    super(ExponentialWithMinLR, self).__init__(optimizer, last_epoch)
    
    if self.warmup_steps:
      print("Using Warmup for Learning: {}".format(warmup_steps))
      self.get_lr()

  def get_lr(self):
    if self.last_epoch == 0:
      return self.base_lrs

    if not self.warmup_steps:
      lr = [group['lr'] * self.gamma for group in self.optimizer.param_groups]
      lr[0] = lr[0] if lr[0] > self.eta_min else self.eta_min
    else:
      local_step = self.optimizer._step_count
      lr = [group['lr'] for group in self.optimizer.param_groups]
      if local_step <= self.warmup_steps + 1:
        lr[0] = self.base_lrs[0] * local_step / self.warmup_steps
        # print("debug:", self.base_lrs[0], local_step / self._warmup_steps, lr[0])
      else:
        lr[0] = lr[0] * self.gamma
        lr[0] = lr[0] if lr[0] > self.eta_min else self.eta_min
    return lr

  def _get_closed_form_lr(self):
    return [base_lr * self.gamma ** self.last_epoch
            for base_lr in self.base_lrs]
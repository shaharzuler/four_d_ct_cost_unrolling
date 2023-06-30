import os
import datetime
from abc import abstractmethod
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler 
from collections import OrderedDict
from torch.utils.data import Dataset

from ..utils.torch_utils import bias_parameters, weight_parameters, load_checkpoint, save_checkpoint


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_set:Dataset, model:torch.nn.Module, losses:dict[str,torch.nn.modules.Module], args:dict): 
        self.train_set = train_set
        self.args = args
        self.model = model
        self.optimizer = self._get_optimizer()
        self.lowest_loss = 1E10
        self.output_root = f"{pathlib.Path(self.args.output_root)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.i_epoch = self.args.after_epoch+1
        self.i_iter = 0
        self.model_suffix = args.model_suffix
        self.loss_modules = losses
        self.scaler = GradScaler()


    def train(self, rank:int, world_size:int) -> None:
        self._init_rank(rank, world_size)
        
        for epoch in range(self.args.epochs):
            break_ = self._run_one_epoch()

            print("Epoch {}".format(self.i_epoch))
            self.i_epoch += 1
            if break_:
                break
        

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    def _init_rank(self, rank:int, world_size:int, update_tensorboard:bool=True) -> None:
        self.world_size = world_size
        self.rank = rank

        if self.rank == 0:
            if update_tensorboard:
                self.summary_writer = SummaryWriter(os.path.join(self.output_root, "summary"))
        
        self.train_loader = self._get_dataloader(self.train_set)

        torch.cuda.set_device(self.rank)
        if self.loss_modules is not None:
            self.loss_modules = {loss_: module_.to(self.rank) for loss_, module_ in self.loss_modules.items()}

        self.model = self._init_model(self.model)

    def _get_dataloader(self, train_set:torch.utils.data.Dataset) -> torch.utils.data.dataloader.DataLoader:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	                    train_set,
                            shuffle=False,
    	                    num_replicas = self.args.n_gpu,
    	                    rank=self.rank)
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_set,
                            batch_size=self.args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            sampler=train_sampler)            
                        
        return train_loader

    def _get_optimizer(self) -> torch.optim.Optimizer:
        param_groups = [
            {'params': bias_parameters(self.model),
             'weight_decay': 0},
            {'params': weight_parameters(self.model),
             'weight_decay': 1e-6}]

        return torch.optim.Adam(param_groups, self.args.lr, betas=(0.9, 0.999), eps=1e-7)

    def _init_model(self, model:torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.rank)
        if self.args.load:
            epoch, weights = load_checkpoint(self.args.load)
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights, strict=False)
        else:
            model.apply(model.init_weights)

        return model

    def _save_model(self, loss:float, name:str) -> None:
        is_best = loss < self.lowest_loss          
        if is_best:
            self.lowest_loss = loss
        
        try:
            models = {'epoch': self.i_epoch, 'state_dict': self.model.module.state_dict()}
        except:
            models = {'epoch': self.i_epoch, 'state_dict': self.model.state_dict()}
        
        save_checkpoint(os.path.join(self.output_root , "checkpoints"), models, name, is_best)
    

    def _deicide_on_early_stop(self) -> bool:
        if self.reduce_loss_delay > self.max_reduce_loss_delay:
            break_ = True
        else:
            break_ = False
        return break_

    def _update_loss_dropping(self, avg_loss:float) -> None:
        if avg_loss < self.lowest_loss:
            self.lowest_loss = avg_loss
            self.reduce_loss_delay = 0
        else:
            self.reduce_loss_delay += 1

import os
import datetime
from abc import abstractmethod
import pathlib
from typing import Dict
from four_d_ct_cost_unrolling.src.utils.os_utils import get_default_checkpoints_path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler 
from collections import OrderedDict
from torch.utils.data import Dataset

from three_d_data_manager import write_config_file

from ..utils.torch_utils import bias_parameters, weight_parameters, load_checkpoint, save_checkpoint


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_set:Dataset, model:torch.nn.Module, losses:Dict[str,torch.nn.modules.Module], args:Dict): 
        self.train_set = train_set
        self.args = args
        self.model = model
        self.optimizer = self._get_optimizer()
        self.lowest_metric_measurement = 1E10
        self.i_epoch = self.args.after_epoch+1
        self.i_iter = 0
        self.epochs_of_metric_not_dropping = 0
        self.model_suffix = args.model_suffix
        self.loss_modules = losses
        self.scaler = GradScaler()

        self.output_root = f"{self.args.output_root}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.args.output_root = self.output_root
        pathlib.Path(self.output_root).mkdir(parents=True, exist_ok=True)
        write_config_file(self.output_root, "training", args)

    def train(self, rank:int) -> str:
        self._init_rank(rank)
        
        for epoch in range(self.args.epochs):
            break_ = self._run_one_epoch()
            self.i_epoch += 1
            if break_:
                break
        
        return self.output_root
        

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    def _init_rank(self, rank:int, update_tensorboard:bool=True) -> None:
        self.rank = rank

        if update_tensorboard:
            self.complete_summary_writer = SummaryWriter(os.path.join(self.output_root, "complete_summary")) 
            self.filtered_summary_writer = SummaryWriter(os.path.join(self.output_root, "filtered_summary")) 
        
        self.train_loader = self._get_dataloader(self.train_set)

        torch.cuda.set_device(self.rank)
        if self.loss_modules is not None:
            self.loss_modules = {loss_: module_.to(self.rank) for loss_, module_ in self.loss_modules.items()}

        self.model = self._init_model(self.model)

    def _get_dataloader(self, train_set:torch.utils.data.Dataset) -> torch.utils.data.dataloader.DataLoader:
        train_sampler = torch.utils.data.distributed.DistributedSampler( #TODO switch to proper sampler
    	                    train_set,
                            shuffle=False,
    	                    num_replicas = self.args.n_gpu,
    	                    rank=0)
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
            if self.args.load == "DEFAULT":
                self.args.load = get_default_checkpoints_path()
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

    def _save_model(self, metric_measurement:float, name:str, save_iter_freq:int) -> None:
        is_best = metric_measurement < self.lowest_metric_measurement          
        if is_best or (self.i_iter % save_iter_freq == 0):
            try:
                models = {'epoch': self.i_epoch, 'state_dict': self.model.module.state_dict()}
            except:
                models = {'epoch': self.i_epoch, 'state_dict': self.model.state_dict()}
            save_checkpoint(os.path.join(self.output_root , "checkpoints"), models, name, is_best) 
    

    def _decide_on_early_stop(self) -> bool:
        if self.epochs_of_metric_not_dropping > self.max_metric_not_dropping_patience:
            break_ = True
            print(f"Early stopping for epochs_of_metric_not_dropping of {self.epochs_of_metric_not_dropping} when max_metric_not_dropping_patience is {self.max_metric_not_dropping_patience}")
        else:
            break_ = False
        return break_

    def _update_metric_dropping(self, metric_measurement:float) -> None:
        if self.i_iter >= self.args.min_save_iter:
            if metric_measurement < self.lowest_metric_measurement:
                self.lowest_metric_measurement = metric_measurement
                self.epochs_of_metric_not_dropping = 0
            else:
                self.epochs_of_metric_not_dropping += 1

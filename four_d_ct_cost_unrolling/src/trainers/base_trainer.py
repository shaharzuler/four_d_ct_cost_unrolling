import torch
import numpy as np
from abc import abstractmethod
from ..utils.torch_utils import bias_parameters, weight_parameters, load_checkpoint, save_checkpoint
import pathlib
from torch.utils.tensorboard import SummaryWriter
# from logger import init_logger
import pprint
from torch.cuda.amp import GradScaler 
from collections import OrderedDict


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_set, model, losses, args):
        self.train_set = train_set
        self.args = args
        self.model = model
        self.optimizer = self._get_optimizer()
        self.lowest_loss = 1E10
        self.save_root = pathlib.Path(self.args.save_root)
        self.i_epoch = self.args.after_epoch+1
        self.i_iter = 0
        self.model_suffix = args.model_suffix
        self.loss_modules = losses
        self.scaler = GradScaler()


    def train(self, rank, world_size):
        self._init_rank(rank,world_size)
        
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

    def _init_rank(self, rank, world_size, update_tensorboard=True):
        self.world_size = world_size
        self.rank = rank

        # init logger
        if self.rank == 0:
            # self._log = init_logger(log_dir=self.args.save_root, filename=self.args.model_suffix + '.log')
            # self._log.info('=> Rank {}: will save everything to {}'.format(self.rank, self.args.save_root))

            # show configurations
            # cfg_str = pprint.pformat(self.args)
            # self._log.info('=> configurations \n ' + cfg_str)
            # self._log.info('{} training samples found'.format(len(self.train_set)))
            # self._log.info('{} validation samples found'.format(len(self.valid_set)))
            if update_tensorboard:
                self.summary_writer = SummaryWriter(str(self.args.save_root))
        
        self.train_loader = self._get_dataloader(self.train_set)
        self.args.epoch_size = min(self.args.epoch_size, len(self.train_loader))

        torch.cuda.set_device(self.rank)
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

    def _get_optimizer(self):
        param_groups = [
            {'params': bias_parameters(self.model),
             'weight_decay': 0},
            {'params': weight_parameters(self.model),
             'weight_decay': 1e-6}]

        return torch.optim.Adam(param_groups, self.args.lr,
                                betas=(0.9, 0.999), eps=1e-7)

    def _init_model(self, model):
        model = model.to(self.rank)
        if self.args.load:
            # if self.rank == 0:
            #     self._log.info(f'Loading model from {self.args.load}')
            epoch, weights = load_checkpoint(self.args.load)

            
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights, strict=False)

        else:
            # if self.rank == 0:
            #     self._log.info("=> Train from scratch")
            model.apply(model.init_weights)

        return model

    def _init_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("There\'s no GPU available on this machine,"
                  "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        return device, n_gpu_use

    def _save_model(self, loss, name):
        is_best = loss < self.lowest_loss          
        if is_best:
            self.lowest_loss = loss
        
        try:
            models = {'epoch': self.i_epoch, 'state_dict': self.model.module.state_dict()}
        except:
            models = {'epoch': self.i_epoch, 'state_dict': self.model.state_dict()}
        
        save_checkpoint(self.save_root, models, name, is_best)
    

    def _decide_on_early_stop(self):
        if self.reduce_loss_delay > self.max_reduce_loss_delay:
            break_ = True
        else:
            break_ = False
        return break_

    def _update_loss_dropping(self, avg_loss):
        if avg_loss < self.lowest_loss:
            self.lowest_loss = avg_loss
            self.reduce_loss_delay = 0
        else:
            self.reduce_loss_delay += 1

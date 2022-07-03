#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:26:18 2022

@author: atreyee
"""
import os
import datetime
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import ExecutionTime, prepare_empty_dir
today = datetime.datetime.now()
log_name = today.strftime("%b") + "_" + today.strftime("%d")
if not os.path.exists(log_name):
    os.mkdir(log_name)


class Trainer:

    """
    Trainer.

    Args:
        n_gpus (int): number of GPUs used for training
        max_epoch (int): number of training epoch
        optimize_method (str):specify the optimization method used to train the student model
        scheduler_method (str, optional): specify how learning rate could be scheduled
        learning_rate (float): learning rate for the optimization method
        n_lr_warm_up_epoch (int): number of epochs for learning rate warm up
        coeff_dict (dict): a dictionary which contains coefficients that will be multipled with different loss values, e.g. knowledge distillation loss, student loss ,etc.
        train_data_loader (torch.utils.data.DataLoader): data loader for the training data set
        valid_data_loaders (dict): data loaders for validation
        model (torch.nn.Module): asr model
    """

    def __init__(self,
                 max_epoch= 10,
                 config= None,
                 resume_from_checkpoint= False,
                 criterion= None,
                 optimize_method= 'adam',
                 scheduler_method='linear_decay',
                 learning_rate= 0.001,
                 n_lr_warm_up_epoch= 10,
                 loss_coeff_dict= None,
                 train_data_loader= None,
                 valid_data_loader= None,
                 model= None
        ):
        super().__init__()

        self.root_dir = Path(config['root_dir']) / log_name
        self.checkpoints_dir = self.root_dir / 'checkpoints'
        self.logs_dir = self.root_dir /'logs'
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume_from_checkpoint)

        self.model = model
        self.optimize_method = optimize_method
        self.scheduler_method = scheduler_method
        self.loss_coeff_dict = loss_coeff_dict
        self.resume = resume_from_checkpoint
        self.start_epoch = 1
        self.epochs = max_epoch
        self.checkpoint_frequency = config['chk_pt_freq']
        self.validation_interval = 1
        self.find_min = True
        self.best_score = np.inf

        # define hyper parameters
        self.max_epoch = max_epoch
        self.lr = learning_rate
        self.n_lr_warm_up_epoch = n_lr_warm_up_epoch
        # define data loaders
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        # initialize optimizer
        self.optimizer, self.scheduler = self.configure_optimizers()


    def _resume_checkpoint(self):
            """Resume experiment from latest checkpoint.

            """
            latest_model_path = self.checkpoints_dir / "latest_model.pt"
            assert latest_model_path.exists(), f"{latest_model_path} does not exist, cannot load checkpoint."

            checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)

            self.start_epoch = checkpoint["epoch"] + 1
            self.best_score = checkpoint["best_score"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.model.load_state_dict(checkpoint["model"], strict = False)

            print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoints to <root_dir>/checkpoints directory, which contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters

        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.pt.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict()
        }

        state_dict["model"] = self.model.cpu().state_dict()

        """
        Notes:
            - latest_model.pt:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc.
            - model_<epoch>.pt:
                The parameters of the model.
            - best_model.pt:
                Like latest_model, but only saved when <is_best> is True.
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.pt").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"generator_{str(epoch).zfill(4)}.pt").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.pt").as_posix())
            # # Save the outputs for best_model
            # test_file_names = [line.strip(line) for line in open("test_files.txt", "r")]
        self.model.cuda()


    def configure_optimizers(self):
        if self.optimize_method == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.scheduler_method == "linear_decay":
            def lr_lambda(current_epoch):
                if current_epoch < self.n_lr_warm_up_epoch:
                    return float(current_epoch+1) / float(max(1, self.n_lr_warm_up_epoch)) # current_epoch+1 to prevent lr=0 in epoch 0
                return max(
                    0.0, float(self.max_epoch - current_epoch) / float(max(1, self.max_epoch - self.n_lr_warm_up_epoch)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.scheduler_method == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=1, eta_min=1e-6)
        else:
            return optimizer

        return optimizer, scheduler

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def _is_best(self, score, find_min=True):
        """Check if the current model is the best model
        """
        if find_min and score <= self.best_score:
            self.best_score = score
            return True
        elif not find_min and score >= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def training_step(self):
        for epoch in range(self.start_epoch, self.epochs + 1):

            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            if self.checkpoint_frequency != 0 and (epoch % self.checkpoint_frequency == 0):
                self._save_checkpoint(epoch)

            # if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                # print(f"[{timer.duration()} seconds] Training is over, Validation is in progress...")

                # self._set_models_to_eval_mode()
                # score = self._validation_epoch(epoch)


                # if self._is_best(score, find_min=self.find_min):
                    # self._save_checkpoint(epoch, is_best=True)

            print(f"[{timer.duration()} seconds] End this epoch.")
            torch.cuda.empty_cache()
            print("GPU " + str(torch.cuda.current_device()) + " current active MB: " + str(torch.cuda.memory_stats()["active_bytes.all.current"] * 1e-6))
            print("GPU " + str(torch.cuda.current_device()) + " current reserved MB: " + str(torch.cuda.memory_stats()["reserved_bytes.all.current"] * 1e-6))

        # for loss_name, loss_val in final_loss_components.items():
        #     self.log(loss_name, loss_val, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log('train_final_loss', final_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log('train_logit_diff', logit_diff, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log('train_prob_diff', prob_diff, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return #final_loss


    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return list(self.valid_data_loaders)[0]

    def get_model(self):
        return self.model


class Seq2SeqTraining(Trainer):

    def __init__(self,
                 n_gpus=0,
                 config=None,
                 max_epoch=100,
                 criterion= None,
                 optimize_method="adam",
                 scheduler_method="linear_decay",
                 learning_rate=0.0001,
                 n_lr_warm_up_epoch=0,
                 loss_coeff_dict=None,
                 resume_from_checkpoint = None,
                 train_data_loader = None,
                 valid_data_loader = None,
                 model = None
        ):

        super(Seq2SeqTraining, self).__init__(max_epoch, config, resume_from_checkpoint, criterion,
                                              optimize_method, scheduler_method, learning_rate,
                                              n_lr_warm_up_epoch, loss_coeff_dict, train_data_loader,
                                              valid_data_loader, model)

        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.criterion = criterion
        self.best_score = np.inf

    def _train_epoch(self, epoch):
        loss_total = 0
        for sample in self.train_data_loader:
            loss, sample, log = self.criterion.forward(sample, self.model)
            loss.backward()
            self.optimizer.step()
            loss_total += loss
            self.scheduler.step()

        print("Loss/Train" + str(loss_total / len(self.train_data_loader)))

    def _validation_epoch(self, epoch):
        loss_total = 0
        for sample in self.valid_data_loader:
            loss, sample, log = self.criterion.forward(sample, self.model)
            loss_total += loss
        return loss_total


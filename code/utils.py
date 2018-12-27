import shutil
import time
import logging
from functools import partial
from pytorch_pretrained_bert.optimization import BertAdam
from fastai.callback import SmoothenValue
from fastprogress.fastprogress import MasterBar, ProgressBar
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch import nn
import torch
import json
from tensorboardX import SummaryWriter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DataWrap:
    path: Union[str, Path]
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: Optional[DataLoader] = None


def compute_avg(inp: List, nums: torch.tensor) -> float:
    "Computes average given list of torch.tensor and numbers corresponding to them"
    return (torch.stack(inp) * nums).sum() / nums.sum()


def good_format_stats(names, stats)->str:
    "Format stats before printing."
    str_stats = []
    for name, stat in zip(names, stats):
        t = str(stat) if isinstance(stat, int) else f'{stat.item():.4f}'
        t += ' ' * (len(name) - len(t))
        str_stats.append(t)
    return '  '.join(str_stats)


@dataclass
class Learner:
    uid: str
    data: DataWrap
    mdl: nn.Module
    loss_fn: nn.Module
    cfg: Dict
    eval_fn: nn.Module
    device: torch.device = torch.device('cuda:0')

    def __post_init__(self):
        "Setup log file, load model if required"
        self.log_keys = ['epochs', 'train_loss',
                         'train_acc', 'valid_loss', 'val_acc']

        self.log_file = Path(self.data.path) / 'logs' / f'{self.uid}.txt'
        self.log_dir = self.log_file.parent / f'{self.uid}'
        self.model_file = Path(self.data.path) / 'models' / f'{self.uid}.pth'
        self.prepare_log_file()

        # Set the number of iterations to 0. Updated in loading if required
        self.num_it = 0
        if self.cfg['resume']:
            self.load_model_dict(
                resume_path=self.cfg['resume_path'], load_opt=self.cfg['load_opt'])
        self.best_met = 0

    def prepare_log_file(self):
        "Prepares the log files depending on arguments"
        if self.log_file.exists():
            if self.cfg['del_existing']:
                logger.info(
                    f'removing existing log with same name {self.log_dir.stem}')
                shutil.rmtree(self.log_dir)
                f = self.log_file.open('w')
            else:
                f = self.log_file.open('a')
        else:
            f = self.log_file.open('w')

        cfgtxt = json.dumps(self.cfg)
        f.write(cfgtxt)
        f.write('\n\n')
        f.write('  '.join(self.log_keys) + '\n')
        f.close()
        # The tensorboard writer
        # self.writer = SummaryWriter(
        #     comment='main_mdl', log_dir=str(self.log_dir))
        # self.writer.add_text('HyperParams', cfgtxt)

    def update_log_file(self, towrite: str):
        "Updates the log files as and when required"
        with self.log_file.open('a') as f:
            f.write(towrite + '\n')

    def validate(self, mb: MasterBar) -> List[torch.tensor]:
        "Validation loop, done after every epoch"
        self.mdl.eval()
        with torch.no_grad():
            val_losses = []
            eval_metrics = []
            nums = []
            for batch in progress_bar(self.data.valid_dl, parent=mb):
                for b in batch.keys():
                    batch[b] = batch[b].to(self.device)
                out = self.mdl(batch)
                loss = self.loss_fn(out, batch)
                metric = self.eval_fn(out, batch)
                val_losses.append(loss.detach().cpu())
                eval_metrics.append(metric.detach().cpu())
                nums.append(batch[next(iter(batch))].shape[0])

            del batch
            nums = torch.tensor(nums).float()
            val_loss = compute_avg(val_losses, nums)
            eval_metric = compute_avg(eval_metrics, nums)
            return val_loss, eval_metric

    def train_epoch(self, mb: MasterBar) -> List[torch.tensor]:
        "One epoch used for training"
        self.mdl.train()
        trn_loss = SmoothenValue(0.9)
        trn_acc = SmoothenValue(0.9)
        for batch in progress_bar(self.data.train_dl, parent=mb):
            # Increment number of iterations
            self.num_it += 1
            for b in batch.keys():
                batch[b] = batch[b].to(self.device)
            self.optimizer.zero_grad()
            out = self.mdl(batch)
            loss = self.loss_fn(out, batch)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            metric = self.eval_fn(out, batch)
            trn_loss.add_value(loss.detach().cpu())
            trn_acc.add_value(metric.detach().cpu())
            mb.child.comment = (
                f'LossB {loss: .4f} | SmLossB {trn_loss.smooth: .4f} | AccB {trn_acc.smooth: .4f}')

        del batch
        self.optimizer.zero_grad()
        return trn_loss.smooth, trn_acc.smooth

    def load_model_dict(self, resume_path: Optional[str] = None, load_opt: bool = False):
        "Load the model and/or optimizer"
        if resume_path == "":
            checkpoint = torch.load(self.model_file.open('rb'))
        else:
            try:
                checkpoint = torch.load(open(resume_path, 'rb'))
            except Exception as e:
                logger.error(
                    f'Some problem with resume path: {resume_path}. Exception raised {e}')
                raise e
        self.mdl.load_state_dict(checkpoint['model_state_dict'])
        if 'num_it' in checkpoint.keys():
            self.num_it = checkpoint['num_it']

        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model_dict(self):
        "Save the model and optimizer"
        checkpoint = {'model_state_dict': self.mdl.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'num_it': self.num_it}
        torch.save(checkpoint, self.model_file.open('wb'))

    def fit(self, epochs: int, lr: float):
        "Main training loop"
        mb = master_bar(range(epochs))
        self.optimizer = self.prepare_optimizer(epochs, lr)
        # self.optimizer = self.opt_fn(lr=lr)
        # Loop over epochs
        mb.write(self.log_keys, table=True)
        exception = False
        st_time = time.time()
        try:
            for epoch in mb:
                train_loss, train_acc = self.train_epoch(mb)
                valid_loss, valid_acc = self.validate(mb)
                # print(f'{val_loss: 0.4f}', f'{eval_metric: 0.4f}')
                to_write = [epoch, train_loss,
                            train_acc, valid_loss, valid_acc]
                mb.write([str(stat) if isinstance(stat, int)
                          else f'{stat:.4f}' for stat in to_write], table=True)
                self.update_log_file(
                    good_format_stats(self.log_keys, to_write))
                if self.best_met < valid_acc:
                    self.best_met = valid_acc
                    self.save_model_dict()
        except Exception as e:
            exception = e
            raise e
        finally:
            end_time = time.time()
            self.update_log_file(
                f'epochs done {epoch}. Exited due to exception {exception}. Total time taken {end_time - st_time: 0.4f}')

            if self.best_met < valid_acc:
                self.save_model_dict()

    def prepare_optimizer(self, epochs: int, lr: float):
        "Prepare the optimizer according to squad example"
        num_train_steps = int(
            len(self.data.train_dl.dataset) / self.cfg['bs'] / 1 * epochs)

        param_optimizer = list(self.mdl.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = set(['bias', 'LayerNorm.bias', 'LayerNorm.weight'])
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        t_total = num_train_steps
        opt = BertAdam(optimizer_grouped_parameters,
                       lr=lr,
                       warmup=0.1,
                       t_total=t_total)
        return opt

    def overfit_batch(self, epochs: int, lr: float):
        "Sanity check to see if model overfits on a batch"
        batch = next(iter(self.data.train_dl))
        for b in batch.keys():
            batch[b] = batch[b].to(self.device)
        self.mdl.train()
        opt = self.prepare_optimizer(epochs, lr)

        for i in range(1000):
            opt.zero_grad()
            out = self.mdl(batch)
            loss = self.loss_fn(out, batch)
            loss.backward()
            opt.step()
            met = self.eval_fn(out, batch)
            print(f'Iter {i} | loss {loss: 0.4f} | acc {met: 0.4f}')

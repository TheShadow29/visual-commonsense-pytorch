import _init_paths
import json
import torch
from typing import List
from dat_loader import get_bert_data, InputFeatures
from model_vcr import get_bert_model, get_loss, get_evalfn
from utils import Learner
from torch import optim
from fastprogress import master_bar, progress_bar
from functools import partial
import logging
import fire

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1)


def main(uid, **kwargs):
    # Both return cuda types
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open('./cfg.json'))
    cfg.update(kwargs)
    # num_cuda_devices = torch.cuda.device_count()
    # cfg['bs'] *= num_cuda_devices
    data_gt = get_bert_data(cfg)
    mdl = get_bert_model(cfg)

    loss_fn = get_loss(cfg)
    eval_fn = get_evalfn(cfg)
    # if num_cuda_devices > 1:
    # mdl = torch.nn.DataParallel(mdl)
    # loss_fn = torch.nn.DataParallel(loss_fn)

    mdl.to(device)
    loss_fn.to(device)

    learn = Learner(uid=uid, data=data_gt, mdl=mdl, loss_fn=loss_fn,
                    eval_fn=eval_fn, device=device, cfg=cfg)
    epochs = cfg['epochs']
    lr = cfg['lr']
    # learn.overfit_batch(epochs, lr)
    learn.fit(epochs=epochs, lr=lr)
    return learn


if __name__ == '__main__':
    learn = fire.Fire(main)

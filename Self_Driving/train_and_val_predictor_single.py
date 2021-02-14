import numpy as np
import torch
import argparse
import os.path as osp
import torch.nn as nn

from models.model_predictor_single_resnet18 import Predictor_single
from dataset.sem_predictor_single_roadline import Sem_predictor_single

import wandb

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate (default: 1e-4)')
argparser.add_argument(
        '--wandb-username',
        type=str,
        help='account username of wandb')
argparser.add_argument(
        '--wandb-project',
        type=str,
        help='project name of wandb')
argparser.add_argument(
        '--dataset-dir',
        type=str,
        help='relative directory of training dataset')
argparser.add_argument(
        '--save-dir',
        type=str,
        help='relative directory to save the weight files')
argparser.add_argument(
        '--train-batch',
        type=int,
        default=64,
        help='batch size for training')
argparser.add_argument(
        '--test-batch',
        type=int,
        default=64,
        help='batch size for validation')
argparser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of epochs to train')
args = argparser.parse_args()

wandb.init(entity=args.wandb_username, project=args.wandb_project)

config = wandb.config
config.batch_size = args.train_batch
config.test_batch_size = args.test_batch
config.epochs = args.num_epochs
config.lr = args.lr
config.log_interval = 10

all_sem_cls = [7]

train_dataset = Sem_predictor_single(args.dataset_dir, all_sem_cls, 'train')
train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=10, pin_memory=True, sampler=None)
test_dataset = Sem_predictor_single(args.dataset_dir, all_sem_cls, 'test')
test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.test_batch_size, shuffle=False,
            num_workers=6, pin_memory=True, sampler=None)

lmbda = lambda epoch: 0.95 if epoch < 10 else 0.9

models = []
optimizers = []
for sem_id in range(13):
    if sem_id not in all_sem_cls:
        models.append(None)
        optimizers.append(None)
    else:
        model = Predictor_single().cuda()
        wandb.watch(model, log='all')
        models.append(model)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        optimizers.append(optimizer)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

def train(epoch, train_loader, models):
    for model in models:
        if model:
            model.train()
    keep_ratio = 0.99
    ave_losses = {}
    for sem_id in range(13):
        ave_losses[sem_id] = 0

    for i, (image, label) in enumerate(train_loader, 1):
        for sem_id in all_sem_cls:
            image_iter = image[sem_id].cuda() # (B, 1, 48, 48)
            target = label[sem_id].cuda() # (B, 1)
            pred_dis = models[sem_id](image_iter).cuda() # (B, 1)
            loss_fn_dis = nn.MSELoss().cuda()
            loss = loss_fn_dis(pred_dis, target)

            optimizers[sem_id].zero_grad()
            loss.backward()
            optimizers[sem_id].step()

            if i == 1:
                ave_losses[sem_id] = loss
            else:
                ave_losses[sem_id] = ave_losses[sem_id] * keep_ratio + loss * (1 - keep_ratio)
                
        if i % 50 == 1:
            print('epoch {}, {}/{}, total_loss={:.4f}'
                .format(epoch, i, len(train_loader), sum(ave_losses.values())))

    for sem_id in all_sem_cls:
        wandb.log({'train loss %02d' % sem_id: ave_losses[sem_id]})

def test(epoch, test_loader, models):
    print('start validation')
    for model in models:
        if model:
            model.eval()

    with torch.no_grad():

        ave_losses = {}
        for sem_id in range(13):
            ave_losses[sem_id] = 0

        example_images = []
        for i, (image, label) in enumerate(test_loader, 1):
            for sem_id in all_sem_cls:
                image_iter = image[sem_id].cuda() # (B, 1, 48, 48)
                target = label[sem_id].cuda() # (B, 1)
                pred_dis = models[sem_id](image_iter).cuda() # (B, 1)
                loss_fn_dis = nn.MSELoss().cuda()
                loss = loss_fn_dis(pred_dis, target)
                if i == 1:
                    ave_losses[sem_id] = loss
                else:
                    ave_losses[sem_id] = ave_losses[sem_id] * (i - 1) / i + loss * 1 / i
                if i == 1:
                    if sem_id == 12:
                        for j in range(len(image[12])):
                            example_images.append(wandb.Image(image[12][j], caption="Pred: {}, Truth: {}".format(pred_dis[j], target[j])))
            print('batch', i, '/', len(test_loader))
        wandb.log({'Examples': example_images})
        for sem_id in all_sem_cls:
            wandb.log({'test loss %02d' % sem_id: ave_losses[sem_id]})

if __name__ == "__main__":
    test(0, test_loader, models)
    for epoch in range(1, config.epochs+1):
        train(epoch, train_loader, models)
        test(epoch, test_loader, models)
        scheduler.step()
        save_path = args.save_dir
        for sem_id in all_sem_cls:
            torch.save(models[sem_id].state_dict(), osp.join(save_path, str(sem_id), 'epoch-%02d.pth' % (epoch)))
            wandb.save(osp.join(save_path, str(sem_id), 'epoch-%02d.pth' % (epoch)))


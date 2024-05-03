import argparse
from cgi import test
import logging
from torchvision import models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pdb
import csv
import time
import os
import sys
from os.path import join

import timm
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

# from lora import LoRA_ViT_timm
# from lora_mod import LoRA_ViT_timm
# from nola import NOLA_ViT_timm
from nola_mlp import NOLAmlp_ViT_timm
from dataloader import load_dataset
# from adapter import Adapter_ViT
from utils.result import ResultCLS
from utils.utils import init, save

weightInfo={
            # "small":"WinKawaks/vit-small-patch16-224",
            "base":"vit_base_patch16_224.orig_in21k_ft_in1k",
            "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
            "base_sam":"vit_base_patch16_224.sam", # 1k
            "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
            "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
            "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
            "base_deit":"deit_base_distilled_patch16_224", # 1k
            "large":"google/vit-large-patch16-224",
            "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
            "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
            "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
            "giant_clip":"vit_giant_patch14_clip_224.laion2b",
            "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
            }


def save2file(acc, loss, best_acc, best_ep, exp, fname):
    """Save(append) accuracy values to a csv file.

    """
    with open(fname, 'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([exp, '{:.2f}'.format(acc), '{:.3f}'.format(loss), 'best: {:.1f}'.format(best_acc), best_ep])


def set_classifier_to_identity(model):
    """Change the number of classes in linear classifier equal to embed size and set the
    linear classifier to identity matrix.

    Alternative approach to obtain the pre_logits output (i.e., input to classifier head).
    Set linear layer weight to identity and bias to zero.
    """
    embed_dim = model.module.embed_dim
    model.module.reset_classifier(num_classes=embed_dim)
    model.module.head.weight.data = torch.eye(embed_dim).cuda()
    model.module.head.bias.data = torch.zeros(embed_dim).cuda()
    # return model


def train(epoch, trainset):
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    # Save backbone features for k-NN evaluation
    save_feats = False
    if save_feats:
        feats_dict = {}
        set_classifier_to_identity(net)

    idx = 0
    for image, label in tqdm(trainset, ncols=60, desc="train", unit="b", leave=None):
        idx += 1
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            pred = net.forward(image)
            if save_feats:
                feats_dict[idx] = [pred, label]
                continue
            loss = loss_func(pred, label)

        if save_feats:
            outfile = 'pretrained_feats_clstoken/feats.pth'
            torch.save(outfile, feats_dict)
            print('Features saved in: ', outfile)
            sys.exit()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss = running_loss + loss.item()
    scheduler.step()

    loss = running_loss / len(trainset)
    logging.info(f"\n\nEPOCH: {epoch}, LOSS : {loss:.3f}, LR: {this_lr:.2e}")
    return loss


@torch.no_grad()
def eval(epoch, testset, datatype='val'):
    result.init()
    net.eval()
    for image, label in tqdm(testset, ncols=60, desc=datatype, unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        with autocast(enabled=True):
            pred = net.forward(image)
            result.eval(label, pred)
    result.print(epoch, datatype)
    return


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train_data_path", type=str, default='../data')
    parser.add_argument("--val_data_path", type=str, default='../data')
    parser.add_argument("--outdir", type=str, default='./exp')
    parser.add_argument("--data_info", type=str, default='data.json')
    parser.add_argument("--annotation", type=str, default='data.csv')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", "-nc", type=int, default=100)
    parser.add_argument("--train_type", "-tt", type=str, default="linear", help="nola, lora, full, linear, adapter")
    parser.add_argument("--rank", "-r", type=int, default=4)
    parser.add_argument("--vit", type=str, default="base")
    parser.add_argument("--data_size", type=float, default="1.0")
    parser.add_argument("--kshot", type=int, default=0,
                        help="use only k-samples per category for training")
    parser.add_argument("--kshot_seed", type=int, default=0,
                        help='seed to use to select kshot samples')

    # NOLA params
    parser.add_argument("--ka", type=int, default=1024,
                        help="number of basis matrices (A) for NOLA")
    parser.add_argument("--kb", type=int, default=1024,
                        help="number of basis matrices (B) for NOLA")

    # Optimization params from deit code
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation of pretrained model only, no training')
    parser.add_argument('--load-weights', action='store_true',
                        help='load pretrained lora/nola weights')
    parser.add_argument('--save-weights', action='store_true',
                        help='save trained lora/nola and fc weights')
    parser.add_argument('--weights', type=str, default=None,
                        help='load pretrained lora/nola weights')

    cfg = parser.parse_args()
    ckpt_path = init(cfg.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(cfg)
    acc_file = '%s/acc_file.csv' % cfg.outdir
    start = time.time()

    if cfg.vit == "base":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif cfg.vit == "tiny":
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    elif cfg.vit == "large":
        model = timm.create_model("vit_large_patch16_224", pretrained=True)
    elif cfg.vit == "huge":
        model = timm.create_model("vit_huge_patch14_224", pretrained=True)
    elif cfg.vit == "large_mae":
        model = timm.create_model("vit_large_patch16_224.mae", pretrained=True)
    elif cfg.vit == "base_mae":
        model = timm.create_model("vit_base_patch16_224.mae", pretrained=True)
    elif cfg.vit == "base_dinov2":
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True)
    elif cfg.vit == "base_dino":
        model = timm.create_model(weightInfo["base_dino"], pretrained=True)
    elif cfg.vit == "large_dino":
        model = timm.create_model("vit_large_patch16_224.dino", pretrained=True)
    elif cfg.vit == "base_sam":
        model = timm.create_model(weightInfo["base_sam"], pretrained=True)
    elif cfg.vit == "base_mill":
        model = timm.create_model(weightInfo["base_mill"], pretrained=True)
    elif cfg.vit == "base_beit":
        model = timm.create_model(weightInfo["base_beit"], pretrained=True)
    elif cfg.vit == "base_clip":
        model = timm.create_model(weightInfo["base_clip"], pretrained=True)
    elif cfg.vit == "base_deit":
        model = timm.create_model(weightInfo["base_deit"], pretrained=True)
    elif cfg.vit == "large_clip":
        model = timm.create_model(weightInfo["large_clip"], pretrained=True)
    elif cfg.vit == "large_beit":
        model = timm.create_model(weightInfo["large_beit"], pretrained=True)
    elif cfg.vit == "huge_clip":
        model = timm.create_model(weightInfo["huge_clip"], pretrained=True)
    elif cfg.vit == "giant_eva":
        model = timm.create_model(weightInfo["giant_eva"], pretrained=True)
    elif cfg.vit == "giant_clip":
        model = timm.create_model(weightInfo["giant_clip"], pretrained=True)
    elif cfg.vit == "giga_clip":
        model = timm.create_model(weightInfo["giga_clip"], pretrained=True)
    else:
        print("Wrong training type")
        exit()

    # Add PEFT module to network
    if cfg.train_type == "lora":
        peft_model = LoRA_ViT_timm(model, r=cfg.rank, num_classes=cfg.num_classes)
        net = peft_model.to(device)
    elif cfg.train_type == "nola":
        # NOLA on QKV matrices of attention
        peft_model = NOLA_ViT_timm(model, r=cfg.rank, num_classes=cfg.num_classes,
                                   ka=cfg.ka, kb=cfg.kb)
        net = peft_model.to(device)
    elif cfg.train_type == "nola_mlp":
        # NOLA on MLP layers
        peft_model = NOLAmlp_ViT_timm(model, r=cfg.rank, num_classes=cfg.num_classes,
                                      ka=cfg.ka, kb=cfg.kb)
        # print(peft_model)
        net = peft_model.to(device)
    elif cfg.train_type == "full":
        # Full fine-tuning
        model.reset_classifier(cfg.num_classes)
        peft_model = model
        net = model.to(device)
    elif cfg.train_type == "linear":
        # Linear / classifier head training
        model.reset_classifier(cfg.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        peft_model = model
        net = model.to(device)
    else:
        print("Wrong training type")
        exit()

    num_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 2 ** 20:.3f}M")
    num_train_params = sum(
        p.numel() for np, p in peft_model.named_parameters() if p.requires_grad and 'head' not in np)
    logging.info(f"PEFT trainable parameters: {num_train_params / 2 ** 20:.3f}M")
    num_cls_params = sum(
        p.numel() for np, p in peft_model.named_parameters() if p.requires_grad and 'head' in np)
    print(f"Classifier trainable parameters: {num_cls_params / 2 ** 20:.3f}M")
    # Load pretrained NOLA weights
    if cfg.eval and cfg.weights is None:
        weights = sorted(os.listdir(cfg.outdir))
        cfg.weights = [item for item in weights if '.safetensors' in item][-1]
        cfg.weights = join(cfg.outdir, cfg.weights)
    if cfg.load_weights or cfg.eval:
        if 'nola' in cfg.train_type:
            net.load_nola_parameters(cfg.weights)
        elif 'lora' in cfg.train_type:
            net.load_lora_parameters(cfg.weights)

    net = torch.nn.DataParallel(net)
    trainset, valset, testset = load_dataset(cfg)
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)
    result = ResultCLS(cfg.num_classes)

    if cfg.eval:
        # Only evaluation, no training
        eval(0, valset, datatype='val')
        logging.info(f"BEST VAL: {result.best_val_result:.3f}, EPOCH: {result.best_epoch:3}")
        sys.exit()

    best_val = 0.
    best_val_ep = 0

    for epoch in range(1, cfg.epochs+1):
        if not cfg.eval:
            loss = train(epoch, trainset)
        if epoch == cfg.epochs:
            # Evaluate on val set
            eval(epoch, valset, datatype='val')
            logging.info(f"BEST VAL: {result.best_val_result:.3f}, EPOCH: {result.best_epoch:3}")
            best_val = result.best_val_result
            best_val_ep = result.best_epoch
            # Save weights of best epoch
            if (result.best_epoch == result.epoch) and not cfg.eval and cfg.save_weights:
                if cfg.train_type == "lora":
                    net.module.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"))
                elif (cfg.train_type == 'nola') or (cfg.train_type == 'nola_mlp'):
                    net.module.save_nola_parameters(ckpt_path.replace(".pt", ".safetensors"))
                elif 'nola' not in cfg.train_type:
                    torch.save(net.state_dict(), ckpt_path.replace(".pt", "_best.pt"))
        print(cfg.outdir)
    end = time.time()
    total = (end - start) / 60.
    logging.info(f"BEST VAL: {best_val:.3f}, EPOCH: {best_val_ep:3}, Time: {total:.1f}min")
    if not cfg.eval:
        save2file(result.acc * 100., loss, best_val*100., best_val_ep, cfg.outdir, acc_file)


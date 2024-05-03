import logging
import os
import time

import numpy as np
import torch
import torch as tc


def read_txt_file(fname):
    """Read data from a text file.

    """
    with open(fname, 'r') as f:
        data = [item.strip() for item in f.readlines()]
    return data


def get_acc_from_file(fname):
    data = read_txt_file(fname)
    if '|  VAL' in data[-2]:
        acc = float(data[-2].split('|')[2]) * 100.
        return acc
    elif '|  VAL' in data[-3]:
        acc = float(data[-3].split('|')[2]) * 100.
        return acc
    else:
        return -1


def normalize_int(x, bits=8):
    """Normalize tensor values to int8/4 range.

    """
    # # convert to (0, xmax)
    # x -= x.min()
    # # convert to (0, 1)
    # x /= (x.max() + 1e-8)
    # # convert to (0, 255) and then to (-128, 127)
    # x = (x * 255) - 128
    if bits == 8:
        val = 127
    elif bits == 4:
        val = 7
    x *= (val / (abs(x).max() + 1e-8))
    return x


def unnormalize_int(x, orig, bits=8):
    """x is normalized int8 version of orig. Unnormalize it to original range

    """
    if bits == 8:
        val = 127
    elif bits == 4:
        val = 7
    x *= (abs(orig).max() / val)
    return x


def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

def init(taskfolder):
    # taskfolder = f"./results/"
    if not os.path.exists(taskfolder):
        os.makedirs(taskfolder)
    datafmt = time.strftime("%Y%m%d_%H%M%S")

    log_dir = f"{taskfolder}/{datafmt}.log"
    initLogging(log_dir)
    ckpt_path = f"{taskfolder}/{datafmt}.pt"
    return ckpt_path


def save(result, net, ckpt_path):
    # Save best model
    
    torch.save(net.state_dict(), ckpt_path.replace(".pt", "_last.pt"))

    logging.info(f"BEST : {result.best_result:.3f}, EPOCH: {(result.best_epoch):3}")
    return
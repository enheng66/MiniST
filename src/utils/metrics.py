import sys
from numpy import inf
import torch

def masked_mse(preds, labels, null_val):
    mask = torch.round(labels)> 0
    mask = mask.float()
    mask_mean = torch.mean(mask)  #
    mask /= mask_mean
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels)**2
    mse = torch.nan_to_num(mask * mse)
    return torch.mean(mse)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    mask = torch.round(labels)> 0
    mask = mask.float()
    mask_mean = torch.mean(mask)  #
    mask /= mask_mean
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mae = torch.abs(preds - labels)
    mae = torch.nan_to_num(mask * mae)
    return torch.mean(mae)



def masked_mape(preds, labels, null_val):
    mask = torch.round(labels)> 0
    mask = mask.float()
    mask_mean = torch.mean(mask)  #
    mask /= mask_mean
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mape = torch.abs((preds - labels) / labels)
    mape = torch.nan_to_num(mask * mape)
    return torch.mean(mape) 



def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse
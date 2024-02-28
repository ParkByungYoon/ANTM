import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torchmetrics.regression import R2Score, \
    SymmetricMeanAbsolutePercentageError


def metric_multiple_timestep(gt, pred):
    mse = F.mse_loss(gt, pred)
    mape = torch.mean(torch.sum(torch.abs((gt - pred) / gt), dim=1))
    r2score = R2Score()
    r2_score = torch.mean(torch.stack(
        [r2score(pred.detach().cpu()[i], gt.detach().cpu()[i]) for i in
         range(len(gt))]))

    ad_smape = SymmetricMeanAbsolutePercentageError()
    smape_adjust = torch.mean(torch.stack(
        [ad_smape(pred.detach().cpu()[i], gt.detach().cpu()[i]) * 0.5 for i
         in range(len(gt))]))
    return mse, mape, r2_score, smape_adjust


def metric_one_timestep(gt, pred):
    mse = F.mse_loss(gt, pred)
    mape = torch.mean(torch.sum(torch.abs((gt - pred) / gt), dim=1))
    # r2score = R2Score()
    # r2_score = torch.mean(torch.stack(
    #     [r2score(pred.detach().cpu()[i], gt.detach().cpu()[i]) for i in
    #      range(len(gt))]))

    ad_smape = SymmetricMeanAbsolutePercentageError()
    smape_adjust = torch.mean(torch.stack(
        [ad_smape(pred.detach().cpu()[i], gt.detach().cpu()[i]) * 0.5 for i
         in range(len(gt))]))
    return mse, mape, smape_adjust


def week52_to_month12(week_sales):
    month_sales = [torch.stack(
        [torch.sum(week_sales[:, 13 * i:13 * i + 4], axis=1),
         torch.sum(week_sales[:, 13 * i + 4:13 * i + 8], axis=1),
         torch.sum(week_sales[:, 13 * i + 8:13 * i + 13], axis=1)], axis=1) for
                   i in range(4)]
    month_sales = torch.stack(month_sales, axis=1).reshape(week_sales.shape[0], -1)

    return month_sales

def week12_to_month3(week_sales):
    month_split = week_sales.split(4, dim=1)
    month_sales = []
    for i in range(len(month_split)):
        month_sale = torch.sum(month_split[i], axis=1)
        month_sales.append(month_sale)
        
    month_sales = torch.stack(month_sales).permute(1,0)

    return month_sales

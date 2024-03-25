import os
import argparse
import wandb
import torch
import pandas as pd
import pickle
import pytorch_lightning as pl
import urllib3
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from datetime import datetime
from model import GTM
from dataset import ZeroShotDataset
from utils.get_features import Get_features

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CURL_CA_BUNDLE'] = ''
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)

def run(args, i):
    wandb.login(key='bb98fa97c43ec61dc45feb0a9c45b3ce6d1353e1', relogin=True, force=True)
    print(args)
    pl.seed_everything(args.seed)

    # Load sales data
    df = pd.read_csv(os.path.join(args.prepo_data_folder, f"item_sale_per_week_{args.sales_total_len}.csv"), index_col="품번")

    # text description load
    text_des = pd.read_excel(os.path.join(args.data_folder, "품번description(텍스트).xlsx"), index_col="품번")

    # Load Google trends
    cat_trend_per_item = pickle.load(
        open(os.path.join(args.prepo_data_folder, "cat_trend_per_item_v3.pkl"), 'rb'))
    fab_trend_per_item = pickle.load(
        open(os.path.join(args.prepo_data_folder, "fab_trend_per_item_v3.pkl"), 'rb'))
    col_trend_per_item = pickle.load(
        open(os.path.join(args.prepo_data_folder, "col_trend_per_item_v4.pkl"), 'rb'))

    df = df.drop(index=['MTPT6102', 'MUPT6102'])

    meta_df = pd.read_csv("/home/smart01/SFLAB/bonbak/data/preprocess/meta_data_ANTM.csv", index_col='item_number')
    
    test_list = pickle.load(open("/home/smart01/SFLAB/su_GTM_t/GTM_T_sanguk/12salesweek_test_item_number296.pkl", 'rb')).drop("MTPT6102")[:]
    train_list = df.index[~df.index.isin(test_list)].drop('JROP328D').drop('JROP328E')[:]
    train_df = df.loc[train_list]
    test_df = df.loc[test_list]

    idx_list = pickle.load(open(f"/home/smart01/SFLAB/bonbak/data/index/index_list{i}", 'rb'))

    train_df = train_df[train_df.index.isin(idx_list)]
    test_df = test_df[test_df.index.isin(idx_list)]

    train_dataset = ZeroShotDataset(args.sales_total_len, args.seq_len,
                                   args.output_dim, train_df,
                                   os.path.join(args.data_folder, "images"),
                                   cat_trend_per_item, fab_trend_per_item, col_trend_per_item,
                                    args.trend_len, args.scaler, meta_df)
    train_loader = train_dataset.get_loader(batch_size=args.batch_size)

    test_dataset = ZeroShotDataset(args.sales_total_len, args.seq_len,
                                   args.output_dim, test_df,
                                   os.path.join(args.data_folder, "images"),
                                   cat_trend_per_item, fab_trend_per_item, col_trend_per_item,
                                   args.trend_len, args.scaler, meta_df, train_dataset.opt_lambda, train=False)
    test_loader = test_dataset.get_loader(batch_size=len(test_df), train=False)

    # Create model
    get_features_class = Get_features(args.trend_len, args.sales_total_len, os.path.join(args.data_folder, "images"),
                                      text_des, cat_trend_per_item, fab_trend_per_item, col_trend_per_item)


    model = GTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        gpu_num=args.gpu_num,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        ahead_step = args.ahead_step,
        teacher_forcing = args.teacher_forcing,
        before_meta = args.before_meta,
        total_scaler = train_dataset.total_scaler,
    )


    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%Y%m%d-%H%M")[2:]
    model_savename = dt_string + '_' + args.model_type

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/' + model_savename,
        filename='---{epoch}---',
        monitor='valid_week_given0_ad_smape',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    wandb_logger = WandbLogger(name=model_savename)

    trainer = pl.Trainer(accelerator="gpu", devices=[args.gpu_num], 
                         max_epochs=args.epochs, check_val_every_n_epoch=2,
                         logger=wandb_logger, callbacks=[checkpoint_callback], 
                         log_every_n_steps=len(train_loader))
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader,)
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=2)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM',
                        help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='ssl_project')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    parser.add_argument('--sales_total_len', type=int, default=12)
    parser.add_argument('--only_4weeks_loss', action='store_true')

    args = parser.parse_args()

    # General arguments
    args.log_dir='log'
    args.seed=21

    # Model specific arguments
    args.use_trends=1
    args.use_img=1
    args.use_text=1
    args.num_trends=3

    # wandb arguments
    args.wandb_entity='bonbak'
    args.wandb_proj='sflab-gtm'
    args.wandb_run='Run1'

    args.use_encoder_mask = False
    args.trend_len = 64  # 52
    args.prepo_data_folder = "/home/smart01/SFLAB/sanguk/mind_br_data_prepro/"
    args.data_folder = "/home/smart01/SFLAB/sanguk/mind_br_data/"
    args.text_embedder = 'klue/bert-base'
    args.sales_total_len = 12 # 52  # 12
    args.seq_len = args.sales_total_len
    args.output_dim = args.sales_total_len
    args.autoregressive = 1
    args.scaler = "standard" # "Minmax"
    args.learning_rate = 0.0001
    args.lead_time = 2
    args.no_scaling = False
    args.ahead_step = 6
    args.val_output_week = 12
    args.val_output_month = 3
    args.num_attn_heads = 8
    args.hidden_dim = 512
    args.embedding_dim = 256
    args.only_4weeks_loss = False # True
    args.num_hidden_layers = 2

    args.model_type = "GTM-Classification"
    args.autoregressive_train = False
    args.teacher_forcing = True
    args.epochs = 100 # 500  # 300  # 150  # 500 # 50 # 300  # 5  # 500  # 100 # 5  # 100
    args.batch_size = 16  # 64
    # args.gpu_num = 0
    args.before_meta = False # True #

    for i in range(12):
        run(args, i)



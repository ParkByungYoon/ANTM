import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from dateutil.relativedelta import relativedelta

from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.timefeatures import time_features

from scipy.stats import boxcox
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ZeroShotDataset():
    def __init__(self, sales_total_len, seq_len, output_dim, data_df, img_root,
                 cat_trend, fab_trend, col_trend, trend_len,
                 scaler, no_scaling, meta_df, qcut_df, opt_lambda=None, train=True):
        self.sales_total_len = sales_total_len
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.data_range = self.sales_total_len - self.seq_len - self.output_dim + 1

        self.data_df = data_df
        self.img_root = img_root
        self.cat_trend = cat_trend
        self.fab_trend = fab_trend
        self.col_trend = col_trend

        self.img_df = meta_df.loc[:,meta_df.columns.str.startswith('img')]
        self.text_df = meta_df.loc[:,meta_df.columns.str.startswith('text')]
        self.meta_df = meta_df.iloc[:,3:26]

        self.trend_len = trend_len
        self.scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()
        self.no_scaling = no_scaling

        
        self.qcut_df = qcut_df
        self.qcut_label_mean = qcut_df.groupby('qcut_label')['sales_mean'].mean().values
        self.qcut_label_median = qcut_df.groupby('qcut_label')['sales_mean'].median().values

        if opt_lambda == None:
            _, self.opt_lambda = boxcox(qcut_df.loc[self.data_df.index, 'sales_mean'])
        else:
            self.opt_lambda = opt_lambda

        self.train = train
        self.past_trend_len = trend_len - sales_total_len
        self.col_imputation = self.color_imputation()

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def color_imputation(self):
        col_trend_df = pd.DataFrame(self.col_trend)
        normal_item_df = col_trend_df.drop(columns=[item  for item in col_trend_df.columns if col_trend_df[item][0] == "multi_color"])
        col_imputation = np.mean([np.mean(normal_item_df[item]) for item in normal_item_df.columns])
        return col_imputation

    def preprocess_data(self):
        data = self.data_df.reset_index(drop=True)

        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        sales, release_dates, ntrends, image_features, text_features, sales_stamps, scalers, real_value_sales, item_numbers_idx = [], [], [], [], [], [], [], [], []
        metas = []
        qcut_labels = []
        target_regs = []

        img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            item_numbers_idx.append(idx)

            idx = self.data_df.iloc[idx]._name

            meta = self.meta_df.loc[idx].values
            metas.append(meta)

            qcut = int(self.qcut_df.loc[idx]['qcut_label'])
            qcut_labels.append(qcut)

            reg = boxcox(self.qcut_df.loc[idx]['sales_mean'], self.opt_lambda)
            target_regs.append(reg)

            sales_stamp = []

            row.index = pd.to_datetime(row.index)

            release_date = row.dropna().sort_index().index[0]
            release_dates.append([release_date.year, release_date.month, release_date.day])

            row = row[release_date:release_date + relativedelta(weeks=self.output_dim-1)].resample('7d').sum().fillna(0)

            time_feature_range = pd.date_range(release_date - relativedelta(weeks=52), release_date + relativedelta(weeks=self.output_dim-1), freq='7d')
            sales_stamp.append(time_features(time_feature_range, freq='w')[0].tolist())
            sales_stamp.append(time_features(time_feature_range, freq='m')[0].tolist())
            sales_stamp.append(time_features(time_feature_range, freq='y')[0].tolist())

            real_value_sale = torch.FloatTensor(np.array(row))
            real_value_sales.append(real_value_sale)

            # sale = np.array(row) / 290
            sale = self.scaler.fit_transform(np.array(row).reshape(-1, 1)).flatten()

            if self.no_scaling:
                sale = np.array(row)

            sale = torch.FloatTensor(sale)

            sales.append(sale)


            scalers.append([self.scaler.mean_, self.scaler.scale_]) if isinstance(self.scaler, StandardScaler) else scalers.append([self.scaler.data_min_, self.scaler.data_range_])

            if idx in self.cat_trend.keys():
                cat_ntrend = np.array(self.cat_trend[idx]).reshape(-1,1)
                if cat_ntrend.shape[0] != 64:
                    cat_ntrend = torch.zeros(self.trend_len, 1)

                fab_ntrend = np.array(self.fab_trend[idx]).reshape(-1,1)
                if fab_ntrend.shape[0] != 64:
                    fab_ntrend = torch.zeros(self.trend_len, 1)

                col_ntrend = np.array(self.col_trend[idx]).reshape(-1,1)
                if self.col_trend[idx] == "multi_color":
                    col_ntrend = torch.zeros(self.trend_len,1)
                else:
                    col_scaler = StandardScaler().fit(col_ntrend[:self.past_trend_len])
                    col_ntrend = col_scaler.transform(col_ntrend)
                if col_ntrend.shape[0] != 64:
                    col_ntrend = torch.zeros(self.trend_len, 1)

                cat_scaler = StandardScaler().fit(cat_ntrend[:self.past_trend_len])
                cat_ntrend = cat_scaler.transform(cat_ntrend)
                fab_scaler = StandardScaler().fit(fab_ntrend[:self.past_trend_len])
                fab_ntrend = fab_scaler.transform(fab_ntrend)
            else:
                col_ntrend = torch.zeros(self.trend_len, 1)
                cat_ntrend = torch.zeros(self.trend_len, 1)
                fab_ntrend = torch.zeros(self.trend_len, 1)


            
            multitrends = torch.stack([torch.FloatTensor(cat_ntrend), torch.FloatTensor(fab_ntrend), torch.FloatTensor(col_ntrend)]).squeeze()    
            img = Image.open(os.path.join(self.img_root, idx + '.png')).convert('RGB')

            word_embeddings = torch.FloatTensor(self.text_df.loc[idx].values)
            text_features.append(word_embeddings)

            # Append them to the lists
            ntrends.append(multitrends)
            img = img_transforms(img)
            image_features.append(img)

            sales_stamp = torch.FloatTensor(sales_stamp)
            sales_stamps.append(sales_stamp)

        # Create tensors for each part of the input/output
        item_sales = torch.stack(sales, dim=0)
        temporal_features = torch.stack(sales_stamps, dim=0)
        ntrends = torch.stack(ntrends, dim=0)
        images = torch.stack(image_features, dim=0)
        texts = torch.stack(text_features, dim=0)
        scalers = torch.FloatTensor(np.array(scalers)).view(-1, 2)

        real_value_sales = torch.stack(real_value_sales, dim=0)

        release_dates = torch.tensor(release_dates)

        item_numbers_idx = torch.tensor(item_numbers_idx)

        meta_data = torch.FloatTensor(metas)
        qcut_labels = torch.LongTensor(qcut_labels)

        target_reg = torch.FloatTensor(target_regs)

        
        return TensorDataset(item_sales, temporal_features, ntrends, images, texts, scalers,
                             real_value_sales, release_dates, item_numbers_idx, meta_data, 
                             qcut_labels, target_reg)

    def get_loader(self, batch_size, train=True):
        print('Starting dataset creation process...')
        data_with_gtrends = self.preprocess_data()
        data_loader = None
        if train:
            data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=False, num_workers=4)
        print('Done.')

        return data_loader

    def __len__(self):
        if self.train:
            return len(self.data_df)
        else:
            return len(self.data_df)
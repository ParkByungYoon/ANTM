import os
from PIL import Image
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class Get_features():
    def __init__(self, trend_len, sales_total_len, img_root, text_des,cat_trend, fab_trend, col_trend):
        self.trend_len = trend_len
        self.sales_total_len = sales_total_len
        self.past_trend_len = trend_len - sales_total_len

        self.img_root = img_root
        self.text_des = text_des
        self.cat_trend = cat_trend
        self.fab_trend = fab_trend
        self.col_trend = col_trend

    def get_several_featrues(self, item_number):
        img = Image.open(os.path.join(self.img_root, item_number + '.png')).convert('RGB')
        text_description = list(self.text_des.loc[item_number])

        try:
            cat_ntrend = np.array(self.cat_trend[item_number]).reshape(-1,1)
            fab_ntrend = np.array(self.fab_trend[item_number]).reshape(-1,1)
            col_ntrend = np.array(self.col_trend[item_number]).reshape(-1,1)

            if self.col_trend[item_number] == "multi_color":
                col_ntrend = np.zeros([self.trend_len, 1])

            else:
                col_scaler = StandardScaler().fit(col_ntrend[:self.past_trend_len])
                col_ntrend = col_scaler.transform(col_ntrend)

            # cat_ntrend = StandardScaler().fit_transform(np.array(cat_ntrend).reshape(-1,1)).flatten()
            # fab_ntrend = StandardScaler().fit_transform(np.array(fab_ntrend).reshape(-1, 1)).flatten()
            # col_ntrend = StandardScaler().fit_transform(np.array(col_ntrend).reshape(-1, 1)).flatten()

            cat_scaler = StandardScaler().fit(cat_ntrend[:self.past_trend_len])
            cat_ntrend = cat_scaler.transform(cat_ntrend)
            fab_scaler = StandardScaler().fit(fab_ntrend[:self.past_trend_len])
            fab_ntrend = fab_scaler.transform(fab_ntrend)
        except:
            cat_ntrend, fab_ntrend, col_ntrend = None, None, None

        return img, text_description, cat_ntrend, fab_ntrend, col_ntrend
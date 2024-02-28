import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from loss import l2_distance_cross_entropy_loss

from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError
from torchmetrics.classification import MultilabelHammingDistance
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
from loss import focal_loss
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
from utils.metric import week12_to_month3
from component import *
import textwrap
from matplotlib import font_manager, rc, gridspec
import urllib3
from scipy.special import inv_boxcox
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

font_path = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

df = pd.read_csv(os.path.join("/home/smart01/SFLAB/sanguk/mind_br_data_prepro/", "item_sale_per_week_12_image_text_nofilter.csv"), index_col="품번")
df = df.drop(index=['MTPT6102', 'MUPT6102'])
test_list = pickle.load(open("/home/smart01/SFLAB/su_GTM_t/GTM_T_sanguk/12salesweek_test_item_number296.pkl", 'rb')).drop("MTPT6102")[:]
train_list = df.index[~df.index.isin(test_list)]
train_df = df.loc[train_list]
test_df = df.loc[test_list]


class GTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads,
                 num_layers, use_text, use_img, \
                 trend_len, num_trends, gpu_num, lr,
                 lead_time, batch_size, no_scaling, ahead_step, val_output_week,
                 val_output_month, only_4weeks_loss, get_features, autoregressive_train, teacher_forcing,
                 before_meta, qcut_label_mean, qcut_label_median, boxcox_opt_lambda, use_encoder_mask=1, autoregressive=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.gpu_num = gpu_num
        self.save_hyperparameters()
        self.trend_len = trend_len
        self.lr = lr

        self.lead_time = lead_time

        self.batch_size = batch_size
        self.no_scaling = no_scaling
        self.ahead_step = ahead_step

        self.val_output_week = val_output_week
        self.val_output_month = val_output_month
        self.only_4weeks_loss = only_4weeks_loss
        self.get_features = get_features
        self.autoregressive_train =autoregressive_train
        self.teacher_forcing = teacher_forcing

        self.boxcox_opt_lambda = boxcox_opt_lambda

        self.before_meta = before_meta
        self.output_num = len(qcut_label_mean) if len(qcut_label_mean) != 2 else 1
        self.qcut_label_mean = torch.FloatTensor(qcut_label_mean).to('cuda:' + str(self.gpu_num))
        self.qcut_label_median = torch.FloatTensor(qcut_label_median).to('cuda:' + str(self.gpu_num))
        self.multi_label = torch.triu(torch.ones(self.output_num, self.output_num)).transpose(0, 1).to('cuda:' + str(self.gpu_num))
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='none')

        # Encoder
        self.dummy_encoder = DummyEmbedder(hidden_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, gpu_num)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim,use_encoder_mask, trend_len, num_trends, gpu_num)
        self.static_feature_encoder = StaticFeatureEncoder(embedding_dim, hidden_dim, use_img, use_text, trend_len, num_layers, before_meta)

        self.label_sales_encoder = LabelSalesNetwork(hidden_dim)

        """decoder first 구성"""
        decoder_layer_first = TransformerDecoderLayer_first(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        self.decoder_first = nn.TransformerDecoder(decoder_layer_first, num_layers)
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim * 4, dropout=0.1)

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=104)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.decoder_fc = nn.Sequential(nn.Linear(hidden_dim, self.output_len if not self.autoregressive else 1), nn.Dropout(0.2))

        self.scale_factor_layer_encoder = Scale_factor_layer_encoder(self.hidden_dim, before_meta, self.output_num, dropout=0.2)
        self.encoder_output_decrease = Encoder_output_decrease(hidden_dim)

        self.acc_metric = BinaryAccuracy().to('cuda:' + str(self.gpu_num))
        self.rec_metric = BinaryRecall().to('cuda:' + str(self.gpu_num))
        self.pre_metric = BinaryPrecision().to('cuda:' + str(self.gpu_num))
        self.f1_metric = BinaryF1Score().to('cuda:' + str(self.gpu_num))



    def _generate_square_subsequent_mask(self, size, masking_start, teacher_forcing=False):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:' + str(self.gpu_num))
        if not teacher_forcing:
            if masking_start:
                for i in range(masking_start, masking_start + self.ahead_step - 1):
                    mask[i + 1] = mask[masking_start]
        return mask


    def _generate_deocder_fisrt_mask(self):
        past_trend_len = self.trend_len - self.output_len
        past_mask = torch.zeros((self.trend_len, past_trend_len))

        upper_mask = torch.zeros(past_trend_len, self.output_len)
        upper_mask = upper_mask.float().masked_fill(upper_mask == 0, float('-inf'))

        mask = (torch.triu(torch.ones(self.output_len, self.output_len)) == 1).transpose(1,0)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        mask = torch.cat([past_mask, torch.cat([upper_mask, mask], axis=0)], axis=1).to('cuda:'+str(self.gpu_num))
        return mask


    def _generate_deocder_se_memory_mask(self):
        past_trend_len = self.trend_len - self.output_len
        past_mask = torch.zeros(self.output_len, past_trend_len)
        past_mask = past_mask.float().masked_fill(past_mask == 0, float(0.0))

        mask = torch.triu(torch.ones(self.output_len, self.output_len)) == 1
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

        mask = (torch.cat([past_mask, mask], axis=1)).to('cuda:'+str(self.gpu_num))
        return mask


    def forward(self, item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data):
        """encoder"""
        # IMAGE X TEXT
        img_encoding = self.image_encoder(images)
        text_encoding = self.text_encoder(texts)
        temporal_encoding = self.dummy_encoder(temporal_features)

        if self.before_meta:
            static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, temporal_encoding[:, 52], meta_data)
        else:
            static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, temporal_encoding[:, 52])


        # Scale Factor (multi-class)
        if self.before_meta:
            pred_scale = self.scale_factor_layer_encoder(static_feature_fusion.unsqueeze(1))
        else:
            # pred_scale = self.scale_factor_layer(static_feature_fusion.unsqueeze(1), meta_data)
            pred_cls, pred_reg = self.scale_factor_layer_encoder(static_feature_fusion.unsqueeze(1), meta_data)


        # Sales Forecasting
        temporal_pos_encoding = self.pos_encoder(temporal_encoding)
        gtrend_emb = self.gtrend_encoder(ntrends.permute(0, 2, 1), temporal_pos_encoding[:, :self.trend_len])
        memory_of_decoder_se = self.decoder_first(tgt=gtrend_emb.permute(1,0,2), memory=static_feature_fusion.permute(1,0,2), tgt_mask=self._generate_deocder_fisrt_mask())

        batch = ntrends.shape[0]

        if item_sales != None:
            label_item_sale = item_sales.unsqueeze(2)
            label_item_sale = self.label_sales_encoder(label_item_sale, temporal_pos_encoding[:,52:])

        if self.autoregressive == 1:
            tgt = torch.zeros(batch, self.output_len, self.hidden_dim).to('cuda:' + str(self.gpu_num))
            tgt[:, 0] = self.encoder_output_decrease(static_feature_fusion).squeeze()

            if item_sales != None:
                tgt[:, 1:] = label_item_sale[:, :-1]
            else:
                pass

            tgt_mask = subsequent_mask
            memory_mask = self._generate_deocder_se_memory_mask()
            decoder_out = self.decoder(tgt.transpose(0, 1), memory_of_decoder_se, tgt_mask, memory_mask)
            forecast = self.decoder_fc(decoder_out.transpose(0, 1)).squeeze()

        return forecast.view(-1, self.output_len), pred_cls, pred_reg

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

    def get_binary_score(self, gt, pred):
        pred = pred.squeeze()
        acc = self.acc_metric(pred, gt)
        rec = self.rec_metric(pred, gt)
        pre = self.pre_metric(pred, gt)
        f1 = self.f1_metric(pred, gt)
        return acc, rec, pre, f1

    def get_score(self, gt, pred_ps, inv_transform=False):
        # r2score = R2Score()
        # r2_score = torch.mean(torch.stack(
        #     [r2score(pred_ps.detach().cpu()[i], gt.detach().cpu()[i]) for i in
        #         range(len(gt))]))
        if inv_transform :
            pred_ps = torch.from_numpy(inv_boxcox(pred_ps, self.boxcox_opt_lambda)).squeeze()

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust = torch.mean(torch.stack(
            [ad_smape(pred_ps.detach().cpu()[i], gt.detach().cpu()[i]) * 0.5
                for i in range(len(gt))]))
        # ad_smape = SymmetricMeanAbsolutePercentageError()

        return smape_adjust
    

    def plot_image_text(self, item_number):
        img, text_description, _, _, _ = self.get_features.get_several_featrues(item_number)
        text_description = textwrap.fill(text_description[0], width=17)
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], )
        plt.subplot(gs[0]).imshow(img)
        plt.title(item_number)
        plt.subplot(gs[1]).text(0.1, 0.4, text_description)
        self.logger.log_image(key=f'img_text_{item_number}', images=[plt])
        plt.show()
        plt.clf()

    
    def plot_gt_pred(self, title, date_range, gt, pred_gs, pred_ps, item_number):
        plt.plot(date_range, gt.detach().cpu(), color='r')
        plt.plot(date_range, pred_gs.detach().cpu(), color='g')
        plt.plot(date_range, pred_ps.detach().cpu(), color='b')
        plt.title(f'{title}_{item_number}')
        plt.legend(['gt', 'pred_gs', 'pred_ps'])
        self.logger.log_image(key=f'{title}_{item_number}', images=[plt])
        plt.show()
        plt.clf()


    def get_3to6step_score(self, batch, batch_idx, phase):
        item_sales, temporal_features, ntrends, images, texts, scalers,\
            real_value_sales, release_dates, item_numbers_idx,\
            meta_data, qcut_label, target_reg = batch

        batch_size = self.batch_size
        plot_idx = [batch_idx * batch_size + i for i in range(len(item_sales)) if (batch_idx * batch_size + i) % 50 == 0]

        """ 주 단위 3to6step 구하기"""
        masking_start = 0
        subsequent_mask = self._generate_square_subsequent_mask(self.output_len, masking_start)
        forecasted_sales, pred_scale_0 = self.forward(item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
        forecasted_sales_0 = forecasted_sales[:, :4]
        pred_scale_0 = pred_scale_0

        subsequent_mask = self._generate_square_subsequent_mask(self.output_len, masking_start)
        forecasted_sales, pred_scale_1 = self.forward(item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
        forecasted_sales_1 = forecasted_sales[:, 4:8]
        pred_scale_1 = pred_scale_1

        masking_start = 6
        subsequent_mask = self._generate_square_subsequent_mask(self.output_len, masking_start)
        forecasted_sales, pred_scale_2 = self.forward(item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
        forecasted_sales_2 = forecasted_sales[:, 8:12]
        pred_scale_2 = pred_scale_2

        forecasted_sales_total = torch.cat([forecasted_sales_0, forecasted_sales_1, forecasted_sales_2], axis=1)

        unscaled_forecasted_sales_0 = torch.stack([forecasted_sales_total[i,:4] + pred_scale_0[i] for i in range(len(forecasted_sales_total))])
        unscaled_forecasted_sales_1 = torch.stack([forecasted_sales_total[i, 4:8] + pred_scale_1[i] for i in range(len(forecasted_sales_total))])
        unscaled_forecasted_sales_2 = torch.stack([forecasted_sales_total[i, 8:12] + pred_scale_2[i] for i in range(len(forecasted_sales_total))])

        unscaled_forecasted_sales_predscale = torch.cat([unscaled_forecasted_sales_0, unscaled_forecasted_sales_1, unscaled_forecasted_sales_2], axis=1)
        unscaled_forecasted_sales_gtscale = torch.stack([forecasted_sales_total[i] * scalers[i, 1] + scalers[i, 0] for i in range(len(forecasted_sales_total))])

        gt = real_value_sales[:, :self.val_output_week]
        pred_gs = unscaled_forecasted_sales_gtscale[:, :self.val_output_week]
        pred_ps = unscaled_forecasted_sales_predscale[:, :self.val_output_week]

        r2_score, smape_adjust = self.get_score(gt, pred_ps)

        self.log(f'{phase}_week_3to6step_r2score', r2_score)
        self.log(f'{phase}_week_3to6step_ad_smape', smape_adjust)

        pred_scales_mean = torch.mean(torch.stack([pred_scale_0, pred_scale_1, pred_scale_2]), axis=0)
        smape_adjust_scale = torch.mean(torch.stack([ad_smape(pred_scales_mean.detach().cpu()[i], scalers[:, 0].detach().cpu()[i]) * 0.5 for i in range(len(scalers))]))

        self.log(f'{phase}_week_3to6step_ad_smape_scale', smape_adjust_scale)

        
        for idx in plot_idx:
            inner_idx = idx - batch_idx * batch_size
            release_dates_p = release_dates.detach().cpu()[inner_idx]
            release_dates_p = datetime.date(release_dates_p[0], release_dates_p[1], release_dates_p[2])
            date_range = pd.date_range(release_dates_p, release_dates_p + relativedelta(weeks=self.val_output_week - 1), freq='7d')
            item_number = train_df.iloc[idx]._name

            self.plot_gt_pred(f'{phase}_week_3to6step', date_range, gt[inner_idx], pred_gs[inner_idx], pred_ps[inner_idx], item_number)
            self.plot_image_text(item_number)

        """3개월 metric 및 PLOT 그리기"""
        
        gt = week12_to_month3(real_value_sales[:, :self.val_output_week])
        pred_gs = week12_to_month3(unscaled_forecasted_sales_gtscale[:, :self.val_output_week])
        pred_ps = week12_to_month3(unscaled_forecasted_sales_predscale[:, :self.val_output_week])

        r2_score, smape_adjust = self.get_score(gt, pred_ps)

        self.log(f'{phase}_month_3to6step_r2score', r2_score)
        self.log(f'{phase}_month_3to6step_ad_smape', smape_adjust)

        for idx in plot_idx:
            inner_idx = idx - batch_idx * batch_size
            release_dates_p = release_dates.detach().cpu()[inner_idx]
            release_dates_p = datetime.date(release_dates_p[0], release_dates_p[1], release_dates_p[2])
            date_range = pd.date_range(release_dates_p,
                                        release_dates_p + relativedelta(
                                            months=self.val_output_month),
                                        freq='m')[:self.val_output_month]
            item_number = train_df.iloc[idx]._name

            self.plot_gt_pred(f'{phase}_month_3to6step', date_range, gt[inner_idx], pred_gs[inner_idx], pred_ps[inner_idx], item_number)


        """accumulation 표현"""

        gt = torch.cumsum(gt, dim=-1)
        pred_gs = torch.cumsum(pred_gs, dim=-1)
        pred_ps = torch.cumsum(pred_ps, dim=-1)

        gt_end = gt[:, -1]
        pred_end_gs = pred_gs[:, -1]
        pred_end_ps = pred_ps[:, -1]

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust = torch.mean(torch.stack([ad_smape(pred_end_ps.detach().cpu()[i], gt_end.detach().cpu()[i]) * 0.5
                for i in range(len(gt_end))]))

        self.log(f'{phase}_month_accum_3to6step_ad_smape', smape_adjust)

        for idx in plot_idx:
            inner_idx = idx - batch_idx * batch_size
            release_dates_p = release_dates.detach().cpu()[inner_idx]
            release_dates_p = datetime.date(release_dates_p[0], release_dates_p[1], release_dates_p[2])
            date_range = pd.date_range(release_dates_p,
                                        release_dates_p + relativedelta(
                                            months=self.val_output_month),
                                        freq='m')[:self.val_output_month]
            item_number = train_df.iloc[idx]._name

            self.plot_gt_pred(f'{phase}_month_accum_3to6step', date_range, gt[inner_idx], pred_gs[inner_idx], pred_ps[inner_idx], item_number)


    def get_given_n_score(self, batch, batch_idx, given_n, phase):
        batch_size = self.batch_size

        item_sales, temporal_features, ntrends, images, texts, scalers,\
            real_value_sales, release_dates, item_numbers_idx,\
            meta_data, qcut_label, target_reg = batch

        masking_start = given_n
        subsequent_mask = self._generate_square_subsequent_mask(self.output_len, masking_start)
        given_gt_weeks = given_n

        item_sales_input = item_sales.clone()
        forecasted_sales_total = torch.zeros(item_sales.shape[0], item_sales.shape[1]).to('cuda:' + str(self.gpu_num))

        for i in range(given_gt_weeks, 12):
            forecasted_sales, pred_cls, pred_reg = self.forward(item_sales_input, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
            item_sales_input[:, i] = forecasted_sales[:, i]
            forecasted_sales_total[:, i] = forecasted_sales[:, i]

            if i == 11:
                if self.output_num == 1:
                    pred_mean = self.qcut_label_mean[(pred_cls>0.5).long()].squeeze()
                    pred_median = self.qcut_label_median[(pred_cls>0.5).long()].squeeze()
                    pred_reg = pred_reg.detach().cpu().numpy()
                else:
                    pred_mean = self.qcut_label_mean[torch.argmax(pred_cls, dim=1)]
                    pred_median = self.qcut_label_median[torch.argmax(pred_cls, dim=1)]
                    pred_reg = pred_reg.detach().cpu().numpy()

        smape_adjust_mean = self.get_score(scalers[:, 0], pred_mean)
        self.log(f'{phase}_week_given{given_n}_ad_smape_mean', smape_adjust_mean)

        smape_adjust_median = self.get_score(scalers[:, 0], pred_median)
        self.log(f'{phase}_week_given{given_n}_ad_smape_median', smape_adjust_median)

        smape_adjust_reg = self.get_score(scalers[:, 0], pred_reg, inv_transform=True)
        self.log(f'{phase}_week_given{given_n}_ad_smape_regression', smape_adjust_reg)

        accuracy, recall, precision, f1_score = self.get_binary_score(qcut_label, pred_cls)
        self.log(f'{phase}_week_given{given_n}_accuracy', accuracy)
        self.log(f'{phase}_week_given{given_n}_recall', recall)
        self.log(f'{phase}_week_given{given_n}_precision', precision)
        self.log(f'{phase}_week_given{given_n}_f1_score', f1_score)


    def get_loss(self, batch, batch_idx, phase):
        item_sales, temporal_features, ntrends, images, texts, scalers,\
            real_value_sales, release_dates, item_numbers_idx,\
            meta_data, qcut_label, target_reg = batch

        # getting 1st time-step loss
        masking_start = 0
        subsequent_mask = self._generate_square_subsequent_mask(self.output_len, masking_start, teacher_forcing=self.teacher_forcing)
        # forecasted_sales, pred_scale = self.forward(item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
        forecasted_sales, pred_cls, pred_reg = self.forward(item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)

        loss_0 = torch.mean(F.mse_loss(item_sales[:, :], forecasted_sales[:, :], reduction='none'))
        if self.output_num == 1:
            # loss_0_scale = F.binary_cross_entropy(torch.sigmoid(pred_scale.squeeze()), qcut_label.float())
            # loss_0_scale = focal_loss(pred_cls.squeeze(), qcut_label.float())
            loss_0_scale = F.binary_cross_entropy(pred_cls.squeeze(), qcut_label.float())
            loss_0_scale += F.mse_loss(pred_reg.squeeze(), target_reg)
        else :
            # distance_0_loss = torch.sqrt(F.mse_loss(self.qcut_label_mean[qcut_label], self.qcut_label_mean[torch.argmax(pred_scale, dim=1)], reduction='mean'))
            # loss_0_scale = F.cross_entropy(pred_scale, qcut_label)
            # loss_0_scale += distance_0_loss*0.1
            # loss_0_scale = F.mse_loss(self.qcut_label_mean[qcut_label], self.qcut_label_mean[torch.argmax(pred_scale, dim=1)], reduction='mean')
            # distance_0_loss = self.multi_label_loss(self.multi_label[torch.argmax(pred_scale, dim=1)], self.multi_label[qcut_label])
            # loss_0_scale = F.cross_entropy(pred_scale, qcut_label)
            # loss_0_scale += distance_0_loss
            # loss_0_scale = self.multi_label_loss(pred_scale, self.multi_label[qcut_label])
            loss_0_scale = l2_distance_cross_entropy_loss(pred_cls, qcut_label)

        loss_stack = loss_0.unsqueeze(0)
        loss_scale_stack = loss_0_scale.unsqueeze(0)

        # stacking nth time-series & scale loss
        for i in range(1, self.output_len - self.ahead_step + 1):
            subsequent_mask = self._generate_square_subsequent_mask(self.output_len, i)
            forecasted_sales, pred_cls, pred_reg = self.forward(item_sales, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
            loss = torch.mean(F.mse_loss(item_sales[:, i:], forecasted_sales[:, i:], reduction='none'))
            if self.output_num == 1:
                # loss_scale = F.binary_cross_entropy(torch.sigmoid(pred_scale.squeeze()), qcut_label.float())
                # loss_scale = focal_loss(pred_cls.squeeze(), qcut_label.float())
                loss_scale = F.binary_cross_entropy(pred_cls.squeeze(), qcut_label.float())
                loss_scale += F.mse_loss(pred_reg.squeeze(), target_reg)
            else :
                # distance_loss = torch.sqrt(F.mse_loss(self.qcut_label_mean[qcut_label], self.qcut_label_mean[torch.argmax(pred_scale, dim=1)], reduction='mean'))
                # loss_scale = F.cross_entropy(pred_scale, qcut_label)
                # loss_scale += distance_loss*0.1
                # loss_scale = F.mse_loss(self.qcut_label_mean[qcut_label], self.qcut_label_mean[torch.argmax(pred_scale, dim=1)], reduction='mean')
                # distance_loss = self.multi_label_loss(self.multi_label[torch.argmax(pred_scale, dim=1)], self.multi_label[qcut_label])
                # loss_scale = F.cross_entropy(pred_scale, qcut_label)
                # loss_scale += distance_loss
                # loss_scale = self.multi_label_loss(pred_scale, self.multi_label[qcut_label])
                loss_scale = l2_distance_cross_entropy_loss(pred_cls, qcut_label)


            loss_stack = torch.cat([loss_stack, loss.unsqueeze(0)])
            loss_scale_stack = torch.cat([loss_scale_stack, loss_scale.unsqueeze(0)])

        weighted = torch.FloatTensor([a / 100 for a in reversed(range(1, self.output_len - self.ahead_step + 2))]).to('cuda:' + str(self.gpu_num))
        weighted_loss = loss_stack * weighted
        
        loss = torch.mean(loss_scale_stack)

        # self.log(f'{phase}_loss_sales', loss)
        # self.log(f'{phase}_loss_sales', torch.mean(weighted_loss))
        self.log(f'{phase}_loss', torch.mean(loss_scale_stack))

        return loss


    def training_step(self, train_batch, batch_idx):
        loss = self.get_loss(train_batch, batch_idx, 'train')
        with torch.no_grad():
            self.get_given_n_score(train_batch, batch_idx, given_n=0, phase='train')

        return loss


    def validation_step(self, valid_batch, batch_idx):
        self.get_loss(valid_batch, batch_idx, 'valid')
        self.get_given_n_score(valid_batch, batch_idx, given_n=0, phase='valid')


    def test_step(self, test_batch, batch_idx):
        batch_size = self.batch_size

        item_sales, temporal_features, ntrends, images, texts, scalers,\
            real_value_sales, release_dates, item_numbers_idx,\
            meta_data, qcut_label, target_reg = test_batch

        masking_start = 0
        subsequent_mask = self._generate_square_subsequent_mask(self.output_len, masking_start)
        given_gt_weeks = 0

        item_sales_input = item_sales.clone()
        forecasted_sales_total = torch.zeros(item_sales.shape[0], item_sales.shape[1]).to('cuda:' + str(self.gpu_num))

        for i in range(given_gt_weeks, 12):
            forecasted_sales, pred_cls, pred_reg = self.forward(item_sales_input, temporal_features, ntrends, images, texts, subsequent_mask, meta_data)
            item_sales_input[:, i] = forecasted_sales[:, i]
            forecasted_sales_total[:, i] = forecasted_sales[:, i]

            if i == 11:
                if self.output_num == 1:
                    pred_reg = pred_reg.detach().cpu().numpy()
                else:
                    pred_reg = pred_reg.detach().cpu().numpy()
        
        cpu_qcut_label = qcut_label.copy().cpu().numpy()
        zero_error = self.get_score(scalers[:, 0][qcut_label==0], pred_reg[cpu_qcut_label==0], inv_transform=True)
        one_error = self.get_score(scalers[:, 0][qcut_label==1], pred_reg[cpu_qcut_label==1], inv_transform=True)
        total_error = self.get_score(scalers[:, 0], pred_reg, inv_transform=True)
        
        print(zero_error, one_error, total_error)

    
    def predict_step(self, test_batch, batch_idx):
        return
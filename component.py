import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=104):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(
                -1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1),
                       y.size(-1))  # (timesteps, samples, output_size)

        return y


class StaticFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, trend_len, num_layers, before_meta, dropout=0.2):
        super(StaticFeatureEncoder, self).__init__()

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text

        self.before_meta = before_meta

        self.concat_features_linear = nn.Linear(1024, hidden_dim)

        self.batchnorm = nn.BatchNorm1d(hidden_dim)

        self.feature_fusion = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, img_encoding, text_encoding, temporal_encoding, meta_data=None):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))

        # Build input
        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(condensed_img)
        if self.use_text == 1:
            decoder_inputs.append(text_encoding)
        if self.before_meta:
            decoder_inputs.append(meta_data)
        decoder_inputs.append(temporal_encoding)

        concat_features = torch.cat(decoder_inputs, dim=1)

        if self.before_meta:
            concat_features = self.linear(concat_features)

        features = self.concat_features_linear(concat_features)
        features = self.batchnorm(features)
        features = self.feature_fusion(features.unsqueeze(2))
        features = self.encoder(features)

        return features

class LabelSalesNetwork(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(LabelSalesNetwork, self).__init__()
        self.sales_net = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
        )

    def forward(self, sales, temporal_feature):

        final = self.sales_net(sales)
        final = final + temporal_feature

        return final


class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len,
                 num_trends, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(
            nn.Linear(num_trends, embedding_dim))
        self.trend_len = trend_len
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def forward(self, gtrends, temporal_feature):
        gtrend_emb = self.input_linear(gtrends)  # 같은시점의 col, cat, fab 들어감
        gtrend_emb = gtrend_emb + temporal_feature
        return gtrend_emb


class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num

    def forward(self, word_embeddings):
        word_embeddings = self.dropout(self.fc(word_embeddings))
        return word_embeddings


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Fine tune resnet
        # for c in list(self.resnet.children())[6:]:
        #     for p in c.parameters():
        #         p.requires_grad = True

    def forward(self, images):
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)

        return out.view(*size).contiguous()  # batch_size, 2048, image_size/32, image_size/32


class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        w, m, y = temporal_features[:, 0].unsqueeze(2), temporal_features[:, 1].unsqueeze(2), temporal_features[:, 2].unsqueeze(2)
        w_emb, m_emb, y_emb = self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([w_emb, m_emb, y_emb], dim=2))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings

class TransformerDecoderLayer_first(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer_first, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer_first, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
        tgt2, attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
        tgt2, attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class EncoderOutputDecrease(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, decoder_m_out):
        out = self.linear(decoder_m_out)
        out = self.relu(out)

        return out


class ScaleFactorLayer(nn.Module):
    def __init__(self, hidden_dim, before_meta, output_num, dropout):
        super().__init__()

        self.output_num = output_num
        self.conv1 = nn.Conv2d(1, 1, 3, stride=2, bias=True)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=2, bias=True)
        self.conv3 = nn.Conv2d(1, 1, 3, stride=2, bias=True)
        self.conv4 = nn.Conv2d(1, 1, 3, stride=2, bias=True)
        self.conv5 = nn.Conv2d(1, 1, 3, stride=2, bias=True)
        self.conv6 = nn.Conv2d(1, 1, 3, stride=2, bias=True)


        self.meta_linear = nn.Linear(5, 49, bias=True)

        self.before_meta = before_meta

        if before_meta:
            self.output_layer = nn.Linear(49, output_num, bias=True)
        else:
            self.output_layer = nn.Linear(98, output_num, bias=True)

        self.activation = nn.GELU()

        self.norm1 = nn.LayerNorm(255)
        self.norm2 = nn.LayerNorm(127)
        self.norm3 = nn.LayerNorm(63)
        self.norm4 = nn.LayerNorm(31)
        self.norm5 = nn.LayerNorm(15)
        self.norm6 = nn.LayerNorm(7)

    def forward(self, encoder_out, meta_data=None):
        out = self.norm1(self.conv1(encoder_out))
        out = self.norm2(self.conv2(out))
        out = self.activation(out)
        out = self.norm3(self.conv3(out))
        out = self.norm4(self.conv4(out))
        out = self.activation(out)
        out = self.norm5(self.conv5(out))
        out = self.norm6(self.conv6(out))
        out = self.activation(out)

        out = out.squeeze().reshape(out.shape[0], -1)

        if self.before_meta:
           pass
        else:
            out_meta = self.activation(self.meta_linear(meta_data))
            concat = torch.cat([out, out_meta], dim=1)

        out = self.output_layer(concat)
        return out
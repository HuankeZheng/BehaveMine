import torch
import torch.nn as nn
import math
import src.decoder as dec

import importlib

importlib.reload(dec)


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_len=5000):
        super(EmbeddingLayer, self).__init__()
        d_model = hidden_dim[0]
        self.token_embedding = nn.Embedding(input_dim[0], d_model)
        self.stage_embedding = nn.Embedding(input_dim[1], d_model)

        self.sensor_embedding_w = SensorEmbedding(input_dim[3], hidden_dim)
        self.sensor_embedding_b = SensorEmbedding(input_dim[3], hidden_dim)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.linear = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        act, stage, sw_padded, sb_padded = x
        sW_lens, sB_lens = lengths

        embed = self.token_embedding(act)
        embed_s = self.stage_embedding(stage)
        embed = embed + embed_s

        embed_sensor_w = self.sensor_embedding_w(sw_padded, sW_lens)
        embed_sensor_b = self.sensor_embedding_b(sb_padded, sB_lens)

        embed = embed + embed_sensor_w

        return self.positional_encoding(embed), embed_sensor_w, embed_sensor_b


class SensorEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SensorEmbedding, self).__init__()
        sensor_hidden_dim = hidden_dim.copy()
        sensor_hidden_dim[-1] = 1
        self.embedding = BaseEmbedding(input_dim, sensor_hidden_dim[0])
        self.decoder = dec.Decoder(sensor_hidden_dim)
        self.bi_lstm = nn.LSTM(sensor_hidden_dim[0], sensor_hidden_dim[0], batch_first=True, bidirectional=True)
        self.fc = nn.Linear(sensor_hidden_dim[0] * 2, sensor_hidden_dim[0])

    def forward(self, x, lengths):
        x = x.to(torch.int)
        b_size = x.size(0)
        x = x.view(-1, x.size(2))
        embed_out = self.embedding(x)

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        dec_out = self.decoder(embed_out, mask=mask)

        dec_out_pack = nn.utils.rnn.pack_padded_sequence(dec_out, lengths, batch_first=True, enforce_sorted=False)
        dec_out, _ = self.bi_lstm(dec_out_pack)
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(dec_out, batch_first=True)

        # convert lengths to index
        lengths = torch.tensor(lengths)
        lengths = lengths.to(dec_out.device)
        lengths = lengths - 1

        batch_size = dec_out.size(0)
        time_step_indices = lengths.view(-1, 1).expand(batch_size, dec_out.size(2)).unsqueeze(1)
        last_outputs = torch.gather(dec_out, 1, time_step_indices).squeeze(1)

        last_outputs = last_outputs.view(b_size, -1, last_outputs.size(1))
        last_outputs = self.fc(last_outputs)

        return last_outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BaseEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, max_len=5000):
        super(BaseEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        x = self.token_embedding(x)
        return self.positional_encoding(x)

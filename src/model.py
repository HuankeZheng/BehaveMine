import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

import src.embedding as emb
import src.decoder as dec

importlib.reload(emb)
importlib.reload(dec)


class BehaveMine(nn.Module):
    def __init__(self, input_dim, b_hidden_dim, s_hidden_dim):
        super(BehaveMine, self).__init__()
        self.embed = emb.EmbeddingLayer(input_dim, s_hidden_dim)

        self.decoder = dec.Decoder(b_hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(b_hidden_dim[0], input_dim[0], bias=False),
            nn.Softmax(dim=2)
        )

    def forward(self, input_data, lengths):
        embed, embed_sensor_w, embed_sensor_b = self.embed(input_data, lengths)

        mask = nn.Transformer.generate_square_subsequent_mask(embed.size(1)).to(embed.device)
        dec_out = self.decoder(embed, mask=mask)

        dec_out = dec_out + embed_sensor_b

        out = self.fc(dec_out)

        return out, dec_out

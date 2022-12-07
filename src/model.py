# MIT License

# Copyright (c) 2019 Kim Seonghyeon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from data import PAD_IDX, MyDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000, dropout: float = 0.2):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_emb = torch.zeros((max_len, emb_size))
        pos_emb[:, 0::2] = torch.sin(pos * den)
        pos_emb[:, 1::2] = torch.cos(pos * den)
        pos_emb = pos_emb.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, token_emb: Tensor):
        return self.dropout(token_emb + self.pos_emb[:token_emb.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2Seq(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size,
                 tgt_vocab_size, dim_feedforward: int = 512, dropout: float = 0.2):
        super(Seq2Seq, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_token_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_token_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_token_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_token_emb(tgt))

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)
    
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_token_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_token_emb(tgt)), memory, tgt_mask)

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()
        self.device = device
        
        self.model_er = BertClassifier()
        self.model_ex = BertClassifier()
        self.model_ir = BertClassifier()

        self.model_er.load_state_dict(torch.load('/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_er.pth'))
        self.model_ex.load_state_dict(torch.load('/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_ex.pth'))
        self.model_ir.load_state_dict(torch.load('/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_ir.pth'))

        self.model_er.to(self.device)
        self.model_ex.to(self.device)
        self.model_ir.to(self.device)

        self.model_er.eval()
        self.model_ex.eval()
        self.model_ir.eval()

    def predict(self, sent):
        final_outputs = []

        for i in range(len(sent)):
            sent_data = MyDataset([sent[i]], [0])
            sent_dataloader = torch.utils.data.DataLoader(sent_data, batch_size=1)

            for sent_input, sent_label in sent_dataloader:
                sent_input_ = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in sent_input]
                mask = sent_input_[0]['attention_mask'].to(self.device)
                input_id = sent_input_[0]['input_ids'].squeeze(1).to(self.device)

                with torch.no_grad():
                    output_er = self.model_er(input_id, mask)
                    output_ex = self.model_ex(input_id, mask)
                    output_ir = self.model_ir(input_id, mask)

            final_output = output_er.argmax(dim=1).item() + output_ex.argmax(dim=1).item() + output_ir.argmax(dim=1).item()
            final_outputs.append(float(final_output))

        return torch.tensor(final_outputs)

from train_estd import *
from data import *
from model import DecoderRNN, EncoderRNN, AttnDecoderRNN
from loss import CosineSimilarity

import torch.nn as nn
from torch import optim
from sentence_transformers import SentenceTransformer

import numpy as np
import math
import time
import random


SOS_token = 0
EOS_token = 1


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

    
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attn=False, max_length=40):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    
    if attn == False:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, pairs_train, input_lang_train, output_lang_train, EOS_token, device, print_every=1000, learning_rate=0.01, attn=False):
    start = time.time()

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair_train(random.choice(pairs_train), input_lang_train, output_lang_train, EOS_token, device)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, attn=True)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d  %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_bias = SentenceTransformer('bert-base-nli-mean-tokens')

    train_act, train_emp, valid_act, valid_emp = process_data(model_bias, other=True)

    input_lang_train, output_lang_train, pairs_train = prepareData('act', 'emp', train_act, train_emp, False)
    input_lang_test, output_lang_test, pairs_test = prepareData('act', 'emp', valid_act, valid_emp, False)
    print(random.choice(pairs_train))
    print(random.choice(pairs_test))

    hidden_size = 256

    encoder = EncoderRNN(input_lang_train.n_words, hidden_size, device).to(device)
    decoder = DecoderRNN(hidden_size, output_lang_train.n_words, device).to(device)
    trainIters(encoder, decoder, 75000, pairs_train, input_lang_train, output_lang_train, EOS_token, device, print_every=5000)
    torch.save(encoder.state_dict(), '[encoder path]')
    torch.save(decoder.state_dict(), '[decoder path]')

    attn_encoder = EncoderRNN(input_lang_train.n_words, hidden_size, device).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang_train.n_words, device, dropout_p=0.1).to(device)
    trainIters(attn_encoder, attn_decoder, 75000, pairs_train, input_lang_train, output_lang_train, EOS_token, device, print_every=5000, attn=True)
    torch.save(attn_encoder.state_dict(), '[encoder path]')
    torch.save(attn_decoder.state_dict(), '[decoder path]')

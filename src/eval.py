import torch

from sentence_transformers import SentenceTransformer

import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from model import *
from data import *
from train_estd import *

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--model', type=str, default="ESTD | seq | seqattn | gpt2 | gpt3")
parser.add_argument('--ablation', type=str, default="all", help='all | noD | noL2')

args = None

SOS_token = 0
EOS_token = 1

result_path_gpt2 = '[GPT model results]'

def main():
    global args

    # do normal parsing
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Similarity model ===== 
    model_bias = SentenceTransformer('bert-base-nli-mean-tokens')

    # ===== Perplexity model ===== 
    GPT_model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium').to(device)
    GPT_tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
    GPT_tokenizer.pad_token = GPT_tokenizer.eos_token
    GPT_model.eval()

    # ===== Empathy lv model =====
    discriminator = Discriminator(device)

    if args.model == "ESTD":
        train_act, train_emp, valid_act, valid_emp = process_data(model_bias, other=False)
        # word -> index
        act_dict, act_total_words = build_dict(train_act, 6000)
        emp_dict, emp_total_words = build_dict(train_emp, 6000)

        emp_bos_idx = emp_dict[BOS]
        emp_eos_idx = emp_dict[EOS]

        # index -> index
        act_dict_rev = {v: k for k, v in act_dict.items()}
        emp_dict_rev = {v: k for k, v in emp_dict.items()}

        test_dataloader = SentencesLoader(valid_act, valid_emp, act_dict, emp_dict, batch_size=20)

        G = Seq2Seq(3, 3, emb_size=256, nhead=8, src_vocab_size=act_total_words, tgt_vocab_size=emp_total_words,
                dim_feedforward=256)
        
        if args.ablation == "all":
            G.load_state_dict(torch.load('[model path]'))
        elif args.ablation == "noD":
            G.load_state_dict(torch.load('[model path]'))
        elif args.ablation == "noL2":
            G.load_state_dict(torch.load('[model path]'))

        discriminator = Discriminator(device)
        G.eval()

        similarity, perplexity, bleu_score, empathy_change = evaluation(test_dataloader, 
                                                                        emp_bos_idx, 
                                                                        emp_dict_rev, 
                                                                        act_dict_rev, 
                                                                        G, 
                                                                        discriminator, 
                                                                        model_bias, 
                                                                        GPT_model, 
                                                                        GPT_tokenizer,
                                                                        None, 
                                                                        None, 
                                                                        None, 
                                                                        None,
                                                                        "ESTD",
                                                                        device,
                                                                        None)                                                                 
    elif args.model == "gpt2":
        train_act, train_emp, valid_act, valid_emp = process_data(model_bias, other=True)
        input_lang_test, output_lang_test, pairs_test = prepareData('act', 'emp', valid_act, valid_emp, False)

        similarity, perplexity, bleu_score, empathy_change = evaluation(None, 
                                                                        None, 
                                                                        None, 
                                                                        None, 
                                                                        None, 
                                                                        discriminator, 
                                                                        model_bias, 
                                                                        GPT_model, 
                                                                        GPT_tokenizer,
                                                                        None,
                                                                        None,
                                                                        None, 
                                                                        None,
                                                                        "gpt2",
                                                                        device,
                                                                        pairs_test)

    else:
        train_act, train_emp, valid_act, valid_emp = process_data(model_bias, other=True)

        input_lang_train, output_lang_train, pairs_train = prepareData('act', 'emp', train_act, train_emp, False)
        input_lang_test, output_lang_test, pairs_test = prepareData('act', 'emp', valid_act, valid_emp, False)
        
        hidden_size = 256

        if args.model == 'seq':
            encoder_path = '[encoder path]'
            decoder_path = '[decoder path]'
            encoder = EncoderRNN(input_lang_train.n_words, hidden_size, device)
            decoder = DecoderRNN(hidden_size, output_lang_train.n_words, device)
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
            encoder.to(device)
            decoder.to(device)
            encoder.eval()
            decoder.eval()

        elif args.model == 'seqattn':
            encoder_path = '[encoder path]'
            decoder_path = '[decoder path]'
            encoder = EncoderRNN(input_lang_train.n_words, hidden_size, device)
            decoder = AttnDecoderRNN(hidden_size, output_lang_train.n_words, device, dropout_p=0.1)
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
            encoder.to(device)
            decoder.to(device)
            encoder.eval()
            decoder.eval()
        
        similarity, perplexity, bleu_score, empathy_change = evaluation(None, 
                                                                        None, 
                                                                        None, 
                                                                        None, 
                                                                        None, 
                                                                        discriminator, 
                                                                        model_bias, 
                                                                        GPT_model, 
                                                                        GPT_tokenizer,
                                                                        encoder,
                                                                        decoder,
                                                                        input_lang_test, 
                                                                        output_lang_test,
                                                                        "seq",
                                                                        device,
                                                                        pairs_test)


    print(f"Similarity: {similarity} | Perplexity: {perplexity} | BLEU: {bleu_score} | Empathy Change: {empathy_change}")


def translate(src, src_mask, emp_bos_idx, Generator, max_len=40):
    memory = Generator.encode(src, src_mask)
    batch_size = src.size(1)
    ys = torch.ones(1, batch_size).fill_(emp_bos_idx).type(torch.long)
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        out = Generator.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = Generator.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
    return ys
 

def perplexity(predicted, GPT_model, GPT_tokenizer, device, MAX_LEN=40):
    print("Calculating perplexity...")

    batch_perplexity = []

    # for i in tqdm(range(len(predicted))):
    BATCH_SIZE = 64

    tokenized_input = GPT_tokenizer.batch_encode_plus(predicted, max_length=MAX_LEN, pad_to_max_length=True, truncation=True)
    
    input_ids = tokenized_input['input_ids'] 
    attention_masks = tokenized_input['attention_mask']

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    data = TensorDataset(input_ids, attention_masks)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size = BATCH_SIZE)

    with torch.no_grad():

        for batch in dataloader:
            b_input = batch[0].to(device)
            b_attn = batch[1].to(device)

            outputs = GPT_model(b_input, attention_mask=b_attn, labels=b_input)
    
            loss, logits = outputs[:2]
            batch_perplexity.append(loss.item())

    return math.exp(np.mean(batch_perplexity))


def similarity(tgt, pred, model_bias, emp_dict_rev=None, act_dict_rev=None):
    print("Calculating similarity...")

    batch_similarities = []

    if args.model == "ESTD":
        sentence_embeddings_act = model_bias.encode(decode_sents(tgt.transpose(0, 1), emp_dict_rev, act_dict_rev))
        sentence_embeddings_emp = model_bias.encode(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev))
    else:
        sentence_embeddings_act = model_bias.encode(tgt)
        sentence_embeddings_emp = model_bias.encode(pred)

    for i in tqdm(range(sentence_embeddings_act.shape[0])):
        sent_similarity = cosine_similarity([sentence_embeddings_act[i]],[sentence_embeddings_emp[i]])
        batch_similarities.append(sent_similarity[0][0])

    return np.mean(np.array(batch_similarities))


def bleu_score(pred, tgt, emp_dict_rev=None, act_dict_rev=None):
    print("Calculating BLEU score...")

    batch_bleu = []

    smoothie = SmoothingFunction().method4

    if args.model == "ESTD":
        length = len(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev))
        for i in tqdm(range(length)):
            sent_bleu = sentence_bleu([decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev)[i]], 
                                    decode_sents(tgt.transpose(0, 1), emp_dict_rev, act_dict_rev)[i], smoothing_function=smoothie)
            batch_bleu.append(sent_bleu)
    else:
        length = len(pred)
        for i in tqdm(range(length)):
            sent_bleu = sentence_bleu([pred[i]], tgt, smoothing_function=smoothie)
            batch_bleu.append(sent_bleu)


    return np.mean(np.array(batch_bleu))


def empathy_change(src, pred, emp_dict_rev, act_dict_rev, model):
    if args.model == "ESTD":
        original_emp = torch.sum(model.predict(decode_sents(src.transpose(0, 1), emp_dict_rev, act_dict_rev, False)))
        generate_emp = torch.sum(model.predict(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev)))
    else:
        original_emp = torch.sum(model.predict(src))
        generate_emp = torch.sum(model.predict(pred))
    change = int(generate_emp) - int(original_emp)
    return change / len(pred)


def evaluate(encoder, decoder, sentence, input_lang_test, output_lang_test, device, max_length=40):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang_test, sentence, EOS_token, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []

        if args.model == 'seq':
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    if topi.item() not in output_lang_test.index2word.keys():
                        continue
                    decoded_words.append(output_lang_test.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            return decoded_words

        else:
            decoder_attentions = torch.zeros(max_length, max_length)
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    if topi.item() not in output_lang_test.index2word.keys():
                        continue
                    decoded_words.append(output_lang_test.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
            return decoded_words, decoder_attentions[:di + 1]


def evaluation(test_dataloader, 
               emp_bos_idx, 
               emp_dict_rev, 
               act_dict_rev,
               model_g, 
               model_d, 
               model_bias, 
               GPT_model, 
               GPT_tokenizer, 
               encoder,
               decoder,
               input_lang_test, 
               output_lang_test,
               type,
               device,
               pairs_test=None):

    similarities = []
    perplexities = []
    bleu_scores = []
    empathylevel = []
    
    if type == "ESTD":
        for _, (src, src_lens, tgt, tgt_lens) in enumerate(tqdm(test_dataloader)):
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            
            pred = translate(src, src_mask, emp_bos_idx, model_g)

            similarities.append(similarity(tgt, pred, model_bias, emp_dict_rev, act_dict_rev))
            perplexities.append(perplexity(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev), GPT_model, GPT_tokenizer, device))
            bleu_scores.append(bleu_score(pred, src, emp_dict_rev, act_dict_rev))
            empathylevel.append(empathy_change(src, pred, emp_dict_rev, act_dict_rev, model_d))

    elif type == "gpt2":
        gpt2_results = pd.read_csv(result_path_gpt2)['result']
        gpt2_results = list(gpt2_results.iloc[:])
        pairs_test_ = pairs_test.copy()
        pairs_test_.pop(1274)
        pairs_test_.pop(1145)

        targets = []
        srcs = []

        for i in range(len(pairs_test_)):
            targets.append(pairs_test_[i][1])
            srcs.append(pairs_test_[i][0])

        similarities.append(similarity(targets, gpt2_results, model_bias))
        perplexities.append(perplexity(gpt2_results, GPT_model, GPT_tokenizer, device))
        bleu_scores.append(bleu_score(gpt2_results, srcs))
        empathylevel.append(empathy_change(srcs, gpt2_results, None, None, model_d))

    else:
        predictions = []
        targets = []
        srcs = []
        for i in range(len(pairs_test)):
            
            if args.model == 'seq':
                output_words = evaluate(encoder, decoder, pairs_test[i][0], input_lang_test, output_lang_test, device)
                pred = ' '.join(output_words)  
            else:
                output_words, _ = evaluate(encoder, decoder, pairs_test[i][0], input_lang_test, output_lang_test, device)
                pred = ' '.join(output_words)
        
            predictions.append(pred)
            targets.append(pairs_test[i][1])
            srcs.append(pairs_test[i][0])
        
        similarities.append(similarity(targets, predictions, model_bias))
        perplexities.append(perplexity(predictions, GPT_model, GPT_tokenizer, device))
        bleu_scores.append(bleu_score(predictions, srcs))
        empathylevel.append(empathy_change(srcs, predictions, None, None, model_d))

    return np.mean(np.array(similarities)), np.mean(np.array(perplexities)), np.mean(np.array(bleu_scores)), np.mean(np.array(empathylevel))


if __name__ == '__main__':
    main()

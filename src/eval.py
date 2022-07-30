import torch

from sentence_transformers import SentenceTransformer

import argparse
import numpy as np

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
parser.add_argument('--modelpath', type=str, default="/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_g.pth")
parser.add_argument('--model', type=str, default="ESTD")

args = None


def main():
    global args
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

    train_act, train_emp, valid_act, valid_emp = process_data(model_bias)

    # word -> index
    act_dict, act_total_words = build_dict(train_act, 6000)
    emp_dict, emp_total_words = build_dict(train_emp, 6000)

    emp_bos_idx = emp_dict[BOS]
    emp_eos_idx = emp_dict[EOS]

    # index -> index
    act_dict_rev = {v: k for k, v in act_dict.items()}
    emp_dict_rev = {v: k for k, v in emp_dict.items()}

    test_dataloader = SentencesLoader(valid_act, valid_emp, act_dict, emp_dict, batch_size=20)

    if args.model == "ESTD":
        G = Seq2Seq(3, 3, emb_size=256, nhead=8, src_vocab_size=act_total_words, tgt_vocab_size=emp_total_words,
                dim_feedforward=256)
        G.load_state_dict(torch.load('/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_g.pth'))
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
                                                                    device)

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
    batch_perplexity = []

    for i in range(len(predicted)):
    
        BATCH_SIZE = 1

        tokenized_input = GPT_tokenizer.batch_encode_plus(predicted[i], max_length=MAX_LEN, padding=True, truncation=True)
        
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


def similarity(tgt, pred, emp_dict_rev, act_dict_rev, model_bias):
    batch_similarities = []

    sentence_embeddings_act = model_bias.encode(decode_sents(tgt.transpose(0, 1), emp_dict_rev, act_dict_rev))
    sentence_embeddings_emp = model_bias.encode(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev))

    for i in range(sentence_embeddings_act.shape[0]):
        sent_similarity = cosine_similarity([sentence_embeddings_act[i]],[sentence_embeddings_emp[i]])
        batch_similarities.append(sent_similarity[0][0])

    return np.mean(np.array(batch_similarities))


def bleu_score(pred, tgt, emp_dict_rev, act_dict_rev):
    batch_bleu = []

    smoothie = SmoothingFunction().method4
    length = len(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev))
    for i in range(length):
        sent_bleu = sentence_bleu([decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev)[i]], 
                                   decode_sents(tgt.transpose(0, 1), emp_dict_rev, act_dict_rev)[i], smoothing_function=smoothie)
        batch_bleu.append(sent_bleu)

    return np.mean(np.array(batch_bleu))


def empathy_change(src, pred, emp_dict_rev, act_dict_rev, model):
    original_emp = torch.mean(model.predict(decode_sents(src.transpose(0, 1), emp_dict_rev, act_dict_rev, False)))
    generate_emp = torch.mean(model.predict(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev)))
    change = int(generate_emp) - int(original_emp)
    return change


def evaluation(test_dataloader, emp_bos_idx, emp_dict_rev, act_dict_rev, model_g, model_d, model_bias, GPT_model, GPT_tokenizer, device):

    similarities = []
    perplexities = []
    bleu_scores = []
    empathylevel = []

    for _, (src, src_lens, tgt, tgt_lens) in enumerate(tqdm(test_dataloader)):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        pred = translate(src, src_mask, emp_bos_idx, model_g)
       
        similarities.append(similarity(tgt, pred, emp_dict_rev, act_dict_rev, model_bias))
        perplexities.append(perplexity(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev), GPT_model, GPT_tokenizer, device))
        bleu_scores.append(bleu_score(pred, tgt, emp_dict_rev, act_dict_rev))
        empathylevel.append(empathy_change(src, pred, emp_dict_rev, act_dict_rev, model_d))
           
    return np.mean(np.array(similarities)), np.mean(np.array(perplexities)), np.mean(np.array(bleu_scores)), np.mean(np.array(empathylevel))


if __name__ == '__main__':
    main()

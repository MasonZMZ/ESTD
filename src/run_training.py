from train_estd import *
from data import *
from model import *
from loss import CosineSimilarity

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

import numpy as np

import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--w1', default=0.25, type=float, help='weight_d')
parser.add_argument('--w2', default=0.05, type=float, help='weight_t')
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--d', default="True", type=str, help="discriminative")
parser.add_argument('--t', default="True", type=str, help='theta')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--batch', type=int, default=20, help='batch size')
parser.add_argument('--path', type=str, default="./path")

args = None

def main():
    global args

    # do normal parsing
    args = parser.parse_args()
    print(args)

    if args.d == 'True':
        discriminative = True
    else:
        discriminative = False
    
    if args.t == 'True':
        theta = True
    else:
        theta = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_bias = SentenceTransformer('/bert-model')

    train_act, valid_act, train_emp, valid_emp = get_data_list()
    
    train_emp_ = []
    valid_emp_ = []
    for i in range(len(train_emp)):
        train_emp_.append(train_emp[i][:-1])
    for i in range(len(valid_emp)):  
        valid_emp_.append(valid_emp[i][:-1])
        
    train_act, train_emp = load_data(train_emp_, train_act)
    valid_act, valid_emp = load_data(valid_emp_, valid_act)
    
    # word -> index
    act_dict, act_total_words = build_dict(train_act, 8000)
    emp_dict, emp_total_words = build_dict(train_emp, 8000)

    emp_bos_idx = emp_dict[BOS]
    emp_eos_idx = emp_dict[EOS]

    print(f"act vocabulary size:{act_total_words}")
    print(f"emp vocabulary size:{emp_total_words}")

    # index -> index
    act_dict_rev = {v: k for k, v in act_dict.items()}
    emp_dict_rev = {v: k for k, v in emp_dict.items()}

    train_dataloader = SentencesLoader(train_act, train_emp, act_dict, emp_dict, batch_size=args.batch)
    test_dataloader = SentencesLoader(valid_act, valid_emp, act_dict, emp_dict, batch_size=args.batch)

    EMB_SIZE = 256
    Generator = Seq2Seq(3, 3, emb_size=EMB_SIZE, nhead=args.nhead, src_vocab_size=act_total_words, tgt_vocab_size=emp_total_words,
                dim_feedforward=EMB_SIZE)
    discriminator = Discriminator(device)
    
    Generator.init()

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    loss_fn_t = CosineSimilarity()
    loss_fn_d = nn.MSELoss()
    optimizer = torch.optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    optimizer_D = torch.optim.Adam(discriminator.parameters())
    
    w_d = args.w1
    w_t = args.w2
    w_s = 1 - w_d * 10 - w_t

    loss = train_epoch(args.epoch, 
                       Generator, 
                       discriminator, 
                       train_dataloader, 
                       test_dataloader, 
                       optimizer_D, 
                       optimizer, 
                       loss_fn, 
                       loss_fn_t, 
                       loss_fn_d, 
                       model_bias,
                       w_d,
                       w_t,
                       w_s,
                       emp_bos_idx,
                       emp_dict_rev,
                       act_dict_rev,
                       device,
                       discriminative=discriminative, 
                       theta=theta)

    torch.save(Generator.state_dict(), f'{args.path}model_d{args.d}_t{args.t}_head{args.nhead}_batch{args.batch}_combinedloss.pth')

if __name__ == '__main__':
    main()

from train_estd import *
from data import *
from model import *
from loss import CosineSimilarity

from sentence_transformers import SentenceTransformer

import numpy as np


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_bias = SentenceTransformer('bert-base-nli-mean-tokens')

    train_act, train_emp, valid_act, valid_emp = process_data(model_bias)

    # word -> index
    act_dict, act_total_words = build_dict(train_act, 6000)
    emp_dict, emp_total_words = build_dict(train_emp, 6000)

    emp_bos_idx = emp_dict[BOS]
    emp_eos_idx = emp_dict[EOS]

    print(f"act vocabulary size:{act_total_words}")
    print(f"emp vocabulary size:{emp_total_words}")

    # index -> index
    act_dict_rev = {v: k for k, v in act_dict.items()}
    emp_dict_rev = {v: k for k, v in emp_dict.items()}

    train_dataloader = SentencesLoader(train_act, train_emp, act_dict, emp_dict, batch_size=20)
    test_dataloader = SentencesLoader(valid_act, valid_emp, act_dict, emp_dict, batch_size=20)

    EMB_SIZE = 256
    Generator = Seq2Seq(3, 3, emb_size=EMB_SIZE, nhead=8, src_vocab_size=act_total_words, tgt_vocab_size=emp_total_words,
                dim_feedforward=EMB_SIZE)
    discriminator = Discriminator(device)
    Generator.init()

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    loss_fn2 = CosineSimilarity()
    loss_fn_d = nn.MSELoss()
    optimizer = torch.optim.Adam(Generator.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer_D = torch.optim.Adam(discriminator.parameters())
    SIGMOD_1 = 0.5
    SIGMOD_2 = 0.5

    Lambda_1 = 1 / SIGMOD_1**2
    Lambda_2 = 1 / SIGMOD_2**2

    Contorl = 2 * np.log(SIGMOD_1) + 2 * np.log(SIGMOD_2)

    loss = train_epoch(10, Generator, discriminator, train_dataloader, test_dataloader, 
                       optimizer_D, optimizer, loss_fn, loss_fn2, loss_fn_d, model_bias,
                       Lambda_1, Lambda_2, Contorl, emp_bos_idx, emp_dict_rev, act_dict_rev,
                       device, discriminative=True)

    torch.save(Generator.state_dict(), '/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_g_no_loss2.pth')
    # torch.save(discriminator.state_dict(), '/home/s4566656/anaconda3/envs/mason/empathy_pretrain/model_d.pth')

from io import open
import torch
from tqdm import tqdm

torch.cuda.set_device(0)

from torch.autograd import Variable
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from data import decode_sents, decode_sents_
from model import create_mask, generate_square_subsequent_mask


def train_epoch(num_epochs, Generator, discriminator, train_dataloader, 
                test_dataloader, optimizer_D, optimizer, loss_fn, loss_fn_t,
                loss_fn_d, model_bert, w_d, w_t, w_s, emp_bos_idx, 
                emp_dict_rev, act_dict_rev, device, discriminative=True, theta=False):

    def pad_shortsent(sents, lens_long):
        sents_padded = sents
        padding = torch.zeros(sents.shape[0], lens_long - sents.shape[1], dtype=torch.int)
        sents_padded = torch.cat((sents_padded, padding), 1)
        return sents_padded

    def translate(src, src_mask, emp_bos_idx, max_len=40):
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
    
    @torch.no_grad()
    def test_translate(emp_bos_idx):
        Generator.eval()
        for i, (src, _, empathy_sent, _) in enumerate(test_dataloader):
            src = src.transpose(0, 1)
            empathy_sent = empathy_sent.transpose(0, 1)
            empathy_input = empathy_sent[:-1, :]

            src_mask, _, _, _ = create_mask(src, empathy_input)
            pred = translate(src, src_mask, emp_bos_idx)
            # print(decode_sents(tgt.transpose(0, 1), emp_dict_rev, act_dict_rev)[-1])
            # print(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev)[-1])
            break

    Generator.train()
    for epoch in tqdm(range(num_epochs)):

        for i, (src, src_lens, empathy_sent, tgt_lens) in enumerate(train_dataloader):
            
            if src.shape[1] < empathy_sent.shape[1]:
                src = pad_shortsent(src, empathy_sent.shape[1])
            else:
                empathy_sent = pad_shortsent(empathy_sent, src.shape[1])

            src = src.transpose(0, 1)
            empathy_sent = empathy_sent.transpose(0, 1)

            empathy_input = empathy_sent[:-1, :]
            empathy_out = empathy_sent[1:, :]

            src_out = src[1:, :]

            src_mask, empathy_mask, src_padding_mask, empathy_padding_mask = create_mask(src, empathy_input)
            
            optimizer.zero_grad()
            generated_sent = Generator.forward(src, 
                                               empathy_input, 
                                               src_mask, 
                                               empathy_mask, 
                                               src_padding_mask, 
                                               empathy_padding_mask, 
                                               src_padding_mask)
            
            pred_sent = translate(src, src_mask, emp_bos_idx, empathy_out.shape[0] - 1)
            orig_sent = src_out

            pred_sent = decode_sents(pred_sent, emp_dict_rev, act_dict_rev) 
            orig_sent = decode_sents(orig_sent, emp_dict_rev, act_dict_rev, False)
            
            # ============= Discriminator =============
            if discriminative:
                empathy_scores = discriminator.predict(pred_sent)

                # 6.0 is the max empathy score
                max_empathy_score = torch.ones(empathy_scores.shape[0], dtype=torch.float)
                max_empathy_score = torch.where(max_empathy_score == 1.0, 6.0, max_empathy_score)
                
                loss_d = loss_fn_d(empathy_scores.reshape(-1, 1), max_empathy_score.reshape(-1, 1)) 
                loss_d = Variable(loss_d, requires_grad=True)

            pred_sent = model_bert.encode(pred_sent)
            orig_sent = model_bert.encode(orig_sent)

            pred_sent = torch.from_numpy(pred_sent)
            orig_sent = torch.from_numpy(orig_sent)

            loss = loss_fn(generated_sent.reshape(-1, generated_sent.shape[-1]), empathy_out.reshape(-1))

            if theta:
                loss_t = loss_fn_t(pred_sent, orig_sent)
                loss_t = Variable(loss_t.mean(), requires_grad=True)

            if theta and discriminative:
                losses = w_s * loss + w_t * loss_t + w_d * loss_d 
                losses.backward() 
                
            elif theta and discriminative == False:
                losses = w_s * loss + (w_t + w_d * 10) * loss_t
                losses.backward()
                
            elif theta == False and discriminative:
                losses = w_s * loss + (w_t * 0.1 + w_d) * loss_d
                losses.backward()
                
            elif theta == False and discriminative == False:
                losses = loss
                losses.backward()

            optimizer.step()
            
    return losses.item()

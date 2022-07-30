from io import open
import torch
from tqdm import tqdm

torch.cuda.set_device(0)

from torch.autograd import Variable
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from data import decode_sents, decode_sents_
from model import create_mask, generate_square_subsequent_mask

def train_epoch(num_epochs, Generator, discriminator, train_dataloader, 
                test_dataloader, optimizer_D, optimizer, loss_fn, loss_fn2,
                loss_fn_d, model_bert, Lambda_1, Lambda_2, Contorl, emp_bos_idx, 
                emp_dict_rev, act_dict_rev, device):


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
        for i, (src, _, tgt, _) in enumerate(test_dataloader):
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            tgt_input = tgt[:-1, :]

            src_mask, _, _, _ = create_mask(src, tgt_input)
            pred = translate(src, src_mask, emp_bos_idx)
            print(decode_sents(src.transpose(0, 1), emp_dict_rev, act_dict_rev, False))
            print(decode_sents(tgt.transpose(0, 1), emp_dict_rev, act_dict_rev)[i])
            print(decode_sents(pred.transpose(0, 1), emp_dict_rev, act_dict_rev)[i])
            break

    Generator.train()
    for epoch in range(num_epochs):
        for i, (src, src_lens, tgt, tgt_lens) in enumerate(tqdm(train_dataloader)):
            optimizer_D.zero_grad()
            src_ = src
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)

            # 去掉最后一个字符, input:[BOS,w1,w2] -> output:[w1,w2,BOS]
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            src_ = [list([int(word) for word in sen]) for sen in src_]
            
            src_ = decode_sents_(src_, emp_dict_rev, act_dict_rev)
            d_real_pred = discriminator.predict(src_)
           
            d_real_label = torch.ones(d_real_pred.shape[0], dtype=torch.float).to(device)
            d_real_label = torch.where(d_real_label == 1.0, 6.0, d_real_label)
            
            d_real_error = loss_fn_d(d_real_pred.reshape(-1, 1).to(device), d_real_label.reshape(-1, 1).to(device))  # sixs = true
            d_real_error = Variable(d_real_error, requires_grad = True)
            d_real_error.backward() # compute/store gradients, but don't change params

            # d_fake_data = Generator(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask).detach()  # detach to avoid training G on these labels

            d_pred_sent = translate(src, src_mask, emp_bos_idx, tgt_out.shape[0] - 1)
            d_pred_sent = decode_sents(d_pred_sent, emp_dict_rev, act_dict_rev)

            d_fake_pred = discriminator.predict(d_pred_sent)
            d_fake_label_ = torch.zeros(d_fake_pred.shape[0], dtype=torch.double).to(device)
            d_fake_error = loss_fn_d(d_fake_pred.reshape(-1, 1).to(device), d_fake_label_.reshape(-1, 1).to(device))  # zeros = fake
            d_fake_error = Variable(d_fake_error, requires_grad = True)
            d_fake_error.backward()

            D_error = d_real_error + d_fake_error
            optimizer_D.step()

            if i % 50 == 0:
                print(f"Discriminator Training: {i + 1}/{len(train_dataloader)} Error_D: {D_error}")

        for i, (src, src_lens, tgt, tgt_lens) in enumerate(tqdm(train_dataloader)):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            if src.shape[1] < tgt.shape[1]:
                src = pad_shortsent(src, tgt.shape[1])
            else:
                tgt = pad_shortsent(tgt, src.shape[1])

            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)

            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            src_out = src[1:, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            g_fake_data = Generator(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            optimizer.zero_grad()

            dg_pred_sent = translate(src, src_mask, emp_bos_idx, tgt_out.shape[0] - 1)
            dg_pred_sent = decode_sents(dg_pred_sent, emp_dict_rev, act_dict_rev)

            dg_fake_pred = discriminator.predict(dg_pred_sent)

            dg_fake_label = torch.ones(dg_fake_pred.shape[0], dtype=torch.float).to(device)
            dg_fake_label = torch.where(dg_fake_label == 1.0, 6.0, dg_fake_label)

            dg_fake_error = loss_fn_d(dg_fake_pred.reshape(-1, 1).to(device), dg_fake_label.reshape(-1, 1).to(device))  # 6.0 is the max empathy score
            dg_fake_error = Variable(dg_fake_error, requires_grad = True)
            dg_fake_error.backward() # compute/store gradients, but don't change params

            pred_sent = translate(src, src_mask, emp_bos_idx, tgt_out.shape[0] - 1)
            orig_sent = src_out

            pred_sent = decode_sents(pred_sent, emp_dict_rev, act_dict_rev)
            orig_sent = decode_sents(orig_sent, emp_dict_rev, act_dict_rev, False)

            pred_sent = model_bert.encode(pred_sent)
            orig_sent = model_bert.encode(orig_sent)

            pred_sent = torch.from_numpy(pred_sent)
            orig_sent = torch.from_numpy(orig_sent)
        
            loss1 = loss_fn(g_fake_data.reshape(-1, g_fake_data.shape[-1]), tgt_out.reshape(-1))
            loss1.backward()

            loss2 = loss_fn2(pred_sent, orig_sent)
            loss2 = Variable(loss2.mean(), requires_grad=True)
            loss2.backward()

            losses = torch.tensor((Lambda_1 * loss1.item() + Lambda_2 * loss2.mean().item() + Contorl))
            losses = Variable(losses, requires_grad=True)
            optimizer.step()

            if i % 100 == 0:
                test_translate(emp_bos_idx)
                print(losses.item())
                print(f"Generator Training: {i + 1}/{len(train_dataloader)}")

    
import pandas as pd  
from sentence_transformers import SentenceTransformer
from data import process_data


def main():
    model_bias = SentenceTransformer('bert-base-nli-mean-tokens')
    train_act, train_emp, valid_act, valid_emp = process_data(model_bias, True)
    
    for i in range(len(train_act)):
        original_sent = train_act[i].copy()
        generate_sent = train_emp[i].copy()
        original_emp = int(torch.sum(discriminator.predict(train_act[i])))
        generate_emp = int(torch.sum(discriminator.predict(train_emp[i])))

        if original_emp > generate_emp:
            train_emp[i] = original_sent
            train_act[i] = generate_sent

        with open(f'./data/data_train.txt', 'a', encoding='utf-8') as fp:
            fp.write(' '.join(train_act[i][1:-1]))
            fp.write(' | ')
            fp.write(' '.join(train_emp[i][1:-1]))
            fp.write('\n')

    for i in range(len(valid_act)):
        original_sent = valid_act[i].copy()
        generate_sent = valid_emp[i].copy()
        original_emp = int(torch.sum(discriminator.predict(valid_act[i])))
        generate_emp = int(torch.sum(discriminator.predict(valid_emp[i])))

        if original_emp > generate_emp:
            valid_emp[i] = original_sent
            valid_act[i] = generate_sent

        with open(f'./data/data_valid.txt', 'a', encoding='utf-8') as fp:
            fp.write(' '.join(valid_act[i][1:-1]))
            fp.write(' | ')
            fp.write(' '.join(valid_emp[i][1:-1]))
            fp.write('\n')

if __name__ == '__main__':
    main()

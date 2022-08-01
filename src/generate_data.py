import pandas as pd  
from sentence_transformers import SentenceTransformer
from data import process_data


def main():
    model_bias = SentenceTransformer('bert-base-nli-mean-tokens')
    train_act, train_emp, valid_act, valid_emp = process_data(model_bias, True)
    dict = {'original': valid_act, 'target': valid_emp}  
    df = pd.DataFrame(dict) 
    df.to_csv('[data path]') 


if __name__ == '__main__':
    main()

from io import open
import re
from collections import Counter
import torch
import nltk
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

BOS = "BOS"  
EOS = "EOS" 

UNK = "UNK"
PAD = "PAD"

PAD_IDX = 0
UNK_IDX = 1

MIN_SIMILARITY = 0.5

dataset = load_dataset("blended_skill_talk")


def del_comma_seq(x):
    x = re.sub('_comma_', ',', x)
    return x


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = re.sub(r" . . . ", " ", sentence)
    sentence = re.sub(r" . . ", " ", sentence)
    sentence = sentence.strip()
    return sentence


def get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


def replace_contractions(text, contractions, contractions_re):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data): 
        self.x_data = x_data
        self.y_data = y_data

        self.length = len(self.x_data)

    def __getitem__(self, index): 
        return (self.x_data[index], self.y_data[index])
    
    def __len__(self): 
        return self.length 

    
def unbias_data(sentences_act, sentences_emp, model):
    print("Prepare BERT pretrained model for data unbias...")
    
    sentence_embeddings_act = model.encode(sentences_act)
    sentence_embeddings_emp = model.encode(sentences_emp)

    emp_list = []
    act_list = []

    for i in tqdm(range(len(sentences_act))):
        similarity = cosine_similarity([sentence_embeddings_act[i]],[sentence_embeddings_emp[i]])
        if similarity[0][0] >= MIN_SIMILARITY:
            act_list.append(sentences_act[i])
            emp_list.append(sentences_emp[i])
    
    return act_list, emp_list


def load_data(emp_list, act_list):
    new_emp_list = []
    new_act_list = []

    num_examples = 0
    print("Loading data...")

    for i in tqdm(range(len(emp_list))):
        new_emp_list.append([BOS] + nltk.word_tokenize(emp_list[i].lower()) + [EOS])
        new_act_list.append([BOS] + nltk.word_tokenize(act_list[i].lower()) + [EOS])
    return new_act_list, new_emp_list


def build_dict(sentences, max_words=6000):
    counter = Counter()
    for sentence in sentences:
        for word in sentence:
            counter[word] += 1
    topn = counter.most_common(max_words)
    total_words = len(topn) + 2
    word_dict = {word[0]: i + 2 for i, word in enumerate(topn)}
    word_dict[PAD] = PAD_IDX
    word_dict[UNK] = UNK_IDX
    return word_dict, total_words


def encode_sentences(sents, word_dict: dict):
    return [[word_dict.get(w, UNK_IDX) for w in s] for s in sents]


def decode_sentences(sents, word_dict_rev: dict):
    sents = sents.numpy()
    return [[word_dict_rev.get(w, UNK) for w in s] for s in sents]


def decode_sentences_(sents, word_dict_rev: dict):
    return [[word_dict_rev.get(w, UNK) for w in s] for s in sents]


def sort_sentences(act_sents, emp_sents):
    idx = sorted(range(len(act_sents)), key=lambda x: len(act_sents[x]))
    return [act_sents[i] for i in idx], [emp_sents[i] for i in idx]


class SentencesLoader:

    def __init__(self, train_act, train_emp, act_dict, emp_dict, batch_size=20, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_act, self.train_emp = train_act, train_emp
        self.act_dict, self.emp_dict = act_dict, emp_dict

        self.sents_act = encode_sentences(self.train_act, self.act_dict)
        self.sents_emp = encode_sentences(self.train_emp, self.emp_dict)
        self.sents_act_lens = [len(v) for v in self.sents_act]
        self.sents_emp_lens = [len(v) for v in self.sents_emp]
        self.sents_act_lens_max = max(self.sents_act_lens)
        self.sents_emp_lens_max = max(self.sents_emp_lens)
        self._batch_index = 0
        self.batch_count = len(self.sents_act) // self.batch_size

    # 按最长的句子补齐短句子
    def pad_sentences(self, sentences):
        lens = torch.LongTensor([len(s) for s in sentences])
        max_len = torch.max(lens)
        result = torch.zeros([lens.size(0), max_len], dtype=torch.long)
        for i, sentence in enumerate(sentences):
            result[i, :lens[i]] = torch.IntTensor(sentence)
        return result, lens

    def get_batch(self, i: int):
        s = i * self.batch_size
        e = (i + 1) * self.batch_size
        x_batch, x_lens = self.pad_sentences(self.sents_act[s:e])
        y_batch, y_lens = self.pad_sentences(self.sents_emp[s:e])
        return x_batch, x_lens, y_batch, y_lens

    def __len__(self):
        return self.batch_count

    def __next__(self):
        if self._batch_index > self.batch_count - 1:
            raise StopIteration()
        r = self.get_batch(self._batch_index)
        self._batch_index += 1

        return r

    def __iter__(self):
        self._batch_index = 0
        return self


def decode_sents(sentences, emp_dict_rev, act_dict_rev, is_emp=True):
    word_dict_rev = emp_dict_rev if is_emp else act_dict_rev
    r = decode_sentences(sentences, word_dict_rev=word_dict_rev)
    decoded_sents = []
    for v in r:
        sent = []
        for x in v:
            if x == EOS:
                break
            if x in [BOS, PAD]:
                continue
            sent.append(x)
        decoded_sents.append(" ".join(sent))
    return decoded_sents


def decode_sents_(sentences, emp_dict_rev, act_dict_rev, is_emp=False):
    word_dict_rev = emp_dict_rev if is_emp else act_dict_rev
    r = decode_sentences_(sentences, word_dict_rev=word_dict_rev)
    decoded_sents = []
    for v in r:
        sent = []
        for x in v:
            if x == EOS:
                break
            if x in [BOS, PAD]:
                continue
            sent.append(x)
        decoded_sents.append(" ".join(sent))
    return decoded_sents


def process_data(model_bias, other=False):
    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    contractions, contractions_re = get_contractions(contraction_dict)

    emp_lines_train = []
    act_lines_train = []

    emp_lines_test = []
    act_lines_test = []

    print('Creating restructure dataset...')

    for i in tqdm(range(len(dataset['train']))):
        for j in range(len(dataset['train'][i]['suggestions']['empathetic_dialogues'])):
            emp_lines_train.append(dataset['train'][i]['suggestions']['empathetic_dialogues'][j])
            act_lines_train.append(dataset['train'][i]['suggestions']['convai2'][j])
            emp_lines_train.append(dataset['train'][i]['suggestions']['empathetic_dialogues'][j])
            act_lines_train.append(dataset['train'][i]['suggestions']['wizard_of_wikipedia'][j])

    for i in tqdm(range(len(dataset['test']))):
        for j in range(len(dataset['test'][i]['suggestions']['empathetic_dialogues'])):
            emp_lines_test.append(dataset['test'][i]['suggestions']['empathetic_dialogues'][j])
            act_lines_test.append(dataset['test'][i]['suggestions']['convai2'][j])
            emp_lines_test.append(dataset['test'][i]['suggestions']['empathetic_dialogues'][j])
            act_lines_test.append(dataset['test'][i]['suggestions']['wizard_of_wikipedia'][j])

    act_train, emp_train = unbias_data(act_lines_train, emp_lines_train, model_bias)
    act_test, emp_test = unbias_data(act_lines_test, emp_lines_test, model_bias)
    print("act train length: ", len(act_train))
    print("act test length: ", len(act_test))

    # Clean Contractions
    emp_train = [replace_contractions(sentence, contractions, contractions_re) for sentence in emp_train]
    act_train = [replace_contractions(sentence, contractions, contractions_re) for sentence in act_train]
    emp_test = [replace_contractions(sentence, contractions, contractions_re) for sentence in emp_test]
    act_test = [replace_contractions(sentence, contractions, contractions_re) for sentence in act_test]

    # proprecessing
    emp_train = [preprocess_sentence(sentence) for sentence in emp_train]
    act_train = [preprocess_sentence(sentence) for sentence in act_train]
    emp_test = [preprocess_sentence(sentence) for sentence in emp_test]
    act_test = [preprocess_sentence(sentence) for sentence in act_test]

    if other:
        return act_train, emp_train, act_test, emp_test

    train_act, train_emp = load_data(emp_train, act_train)
    valid_act, valid_emp = load_data(emp_test, act_test)

    return train_act, train_emp, valid_act, valid_emp

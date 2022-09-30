# ESTD: Empathy Style Transformer with Discriminative mechanism
[ADMA 2022] Mingzhe Zhang, Lin Yue, Miao Xu*

# Overview
we employ ESTD to transfer a sentence from the source to the target text(higher empathy level). We believe our method and findings are a crucial step in establishing a friendly and inclusive online communication environment while furthering the development of a mental health platform. 

![structure](https://github.com/MasonZMZ/ESTD/blob/main/img/newStructure.png)
# Set-up

## Operation System:
![macOS Badge](https://img.shields.io/badge/-macOS-white?style=flat-square&logo=macOS&logoColor=000000) ![Linux Badge](https://img.shields.io/badge/-Linux-white?style=flat-square&logo=Linux&logoColor=FCC624) ![Ubuntu Badge](https://img.shields.io/badge/-Ubuntu-white?style=flat-square&logo=Ubuntu&logoColor=E95420)

## Requirements:
![Python](http://img.shields.io/badge/-3.8.13-eee?style=flat&logo=Python&logoColor=3776AB&label=Python) ![PyTorch](http://img.shields.io/badge/-1.12.0-eee?style=flat&logo=pytorch&logoColor=EE4C2C&label=PyTorch) ![Scikit-learn](http://img.shields.io/badge/-1.1.1-eee?style=flat&logo=scikit-learn&logoColor=e26d00&label=Scikit-Learn) ![NumPy](http://img.shields.io/badge/-1.22.3-eee?style=flat&logo=NumPy&logoColor=013243&label=NumPy) ![tqdm](http://img.shields.io/badge/-4.64.0-eee?style=flat&logo=tqdm&logoColor=FFC107&label=tqdm) ![pandas](http://img.shields.io/badge/-1.4.3-eee?style=flat&logo=pandas&logoColor=150458&label=pandas) ![SciPy](http://img.shields.io/badge/-1.8.1-eee?style=flat&logo=SciPy&logoColor=8CAAE6&label=SciPy) ![colorama](http://img.shields.io/badge/-0.4.5-eee?style=flat&label=colorama) ![cudatoolkit](http://img.shields.io/badge/-11.6.0-eee?style=flat&label=cudatoolkit) ![datasets](http://img.shields.io/badge/-2.4.0-eee?style=flat&label=datasets) ![matplotlib](http://img.shields.io/badge/-3.4.2-eee?style=flat&label=matplotlib) ![nltk](http://img.shields.io/badge/-3.7-eee?style=flat&label=nltk) ![tokenizers](http://img.shields.io/badge/-0.11.4-eee?style=flat&label=tokenizers) ![transformers](http://img.shields.io/badge/-4.18.0-eee?style=flat&label=transformers) ![seaborn](http://img.shields.io/badge/-0.11.2-eee?style=flat&label=seaborn)

## GPU:

![Nvidia](http://img.shields.io/badge/-RTX_A6000_48GB-eee?style=flat&logo=NVIDIA&logoColor=76B900&label=NVIDIA)

## Environment
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install transformers
pip install datasets
```
## Preparing Data

```
from datasets import load_dataset
dataset = load_dataset("blended_skill_talk")
```

# Evaluation
## Evaluation Code
```
python eval.py --gpu 0 --modelpath [model_path] --model ESTD
```

# Training
```
python /src/run_training.py
```

## Reproduce
| Original Utterance | Rewritten Utterance |
| ---- | ---- | 
| Well, you better figure out how to fix it.  | I'am sorry to hear that. It will be tough time.| 
| Oh, just a nail? You are a nice person.     | I'am sorry to hear that. How do you feel? | 
| Some here would love to go.                 | I like to get my free time too. You can to me, lol.| 

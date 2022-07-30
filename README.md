# ESTD: Empathy Style Transformer with Discriminative mechanism
[ADMA 2022] Mingzhe Zhang*, Lin Yue, Miao Xu

# Overview
we employ ESTD to transfer a sentence from the source to the target text(higher empathy level). 

![Structure](./img/structure.pdf)
# Set-up
## Environment
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install transformers
```
## Preparing Data
pip install datasets

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

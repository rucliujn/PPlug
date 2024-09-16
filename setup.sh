#!/bin/bash
set -x

# 1.13.1
# use python 3.9
alias python=python3.9
alias pip=pip3.9

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install accelerate
pip install peft

pip install pybind11==2.10.3
pip install nltk==3.8.1
pip install Wandb==0.13.11
pip install Ninja==1.11.1
pip install shutil

pip install lm_eval>=0.3.0
pip install numpy>=1.22.0
pip install pybind11>=2.6.2
pip install regex
pip install sentencepiece
pip install six
pip install tiktoken>=0.1.2
pip install tokenizers>=0.12.1
pip install transformers>=4.38.0
pip install evaluate
pip install deepspeed==0.13.1
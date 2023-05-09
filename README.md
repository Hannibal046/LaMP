# LaMP
This repo contains unofficial reproduction of [LaMP benchmark](https://lamp-benchmark.github.io) focusing on **_When Large Language Models Meet Personalization_**.

## Progress
- [ ] BERTScore for answer filtering
- [ ] LLM inference
- [ ] bm25 retrieval
- [x] finetune scrips
- [x] read paper and download data

## Requirement

```
pip install transformers openai ipywidgets ipykernel sacrebleu nltk pynvml editdistance tiktoken rouge-metric compare-mt accelerate wandb
pip install hydra-core --upgrade
pip install -e .
```


## Data
Download data using `bash data/download.sh`

## Retrieval
Retrieval is based on `Pyserini`.

BM25 retrieval:`python notebooks/bm25.py`

## FineTune
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 lamp/main.py
```
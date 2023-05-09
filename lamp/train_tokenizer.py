from tokenizers import Tokenizer,SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing
import transformers
import argparse
import json

def get_data_iterator(data_path):
    ret = []
    batch_size = 10_000
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            for v in data.values():
                ret.append(v)
            
            if len(ret)>=batch_size:
                    yield ret
                    ret = []    
    if ret:
        yield ret
                    
jrc_ende_config = {
    "data_path":"../data/jrc_ende/train.jsonl",
    "output_dir":"../data/jrc_ende/tokenizer",
    "src":"src",
    "trg":"trg",
    "vocab_size":32000,
}

jrc_enes_config = {
    "data_path":"../data/jrc_enes/train.jsonl",
    "output_dir":"../data/jrc_enes/tokenizer",
    "src":"en",
    "trg":"es",
    "vocab_size":32000,
}

jrc_esen_config = {
    "data_path":"../data/jrc_esen/train.jsonl",
    "output_dir":"../data/jrc_esen/tokenizer",
    "src":"es",
    "trg":"en",
    "vocab_size":32000,
}

jrc_deen_config = {
    "data_path":"../data/jrc_deen/train.jsonl",
    "output_dir":"../data/jrc_deen/tokenizer",
    "src":"de",
    "trg":"en",
    "vocab_size":32000,
}

# wmt_ende_config = {
#     "data_path":"../data/wmt/ende/train.jsonl",
#     "output_dir":"../data/wmt/ende/tokenizer",
#     "src":"en",
#     "trg":"de",
#     "vocab_size":36_000,
# }

# wmt_enzh_config = {
#     "data_path":"../data/wmt/enzh/train.jsonl",
#     "output_dir":"../data/wmt/enzh/tokenizer",
#     "src":"en",
#     "trg":"zh",
#     "vocab_size":36_000,
# }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path",required=True)
    parser.add_argument("--output_path",required=True)
    parser.add_argument("--vocab_size",default=32_000,type=int)
    args = parser.parse_args()
    
    data_iterator = get_data_iterator(args.train_data_path)

    special_tokens = ['<bos>', '<eos>', '<unk>', '<sep>', '<pad>', '<cls>', '<mask>']
    bos,eos,unk,sep,pad,cls,mask = special_tokens

    _tokenizer = SentencePieceBPETokenizer()
    _tokenizer.train_from_iterator(
        data_iterator,
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens
    )
    _tokenizer.post_processor = TemplateProcessing(
        single=f"$A {eos}",
        pair=f"$A {sep} $B:1 {eos}:1",
        special_tokens=[
            (eos, _tokenizer.token_to_id(eos)),
            (sep, _tokenizer.token_to_id(sep)),
        ],
    )
    # convert
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=_tokenizer, model_max_length=512, special_tokens=special_tokens)

    tokenizer.bos_token = bos
    tokenizer.bos_token_id = _tokenizer.token_to_id(bos)

    tokenizer.pad_token = pad
    tokenizer.pad_token_id = _tokenizer.token_to_id(pad)

    tokenizer.eos_token = eos
    tokenizer.eos_token_id = _tokenizer.token_to_id(eos)

    tokenizer.unk_token =unk
    tokenizer.unk_token_id = _tokenizer.token_to_id(unk)

    tokenizer.cls_token =cls
    tokenizer.cls_token_id = _tokenizer.token_to_id(cls)

    tokenizer.sep_token =sep
    tokenizer.sep_token_id = _tokenizer.token_to_id(sep)

    tokenizer.mask_token = mask
    tokenizer.mask_token_id = _tokenizer.token_to_id(mask)

    tokenizer.save_pretrained(args.output_path)

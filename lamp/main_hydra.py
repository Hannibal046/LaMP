# Built-in Package
from typing import Any
from functools import partial
import os,json,time,random
os.environ["TOKENIZERS_PARALLELISM"]='true'
from copy import copy
import math
import logging
# Third-Partys Package
## Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import Adafactor, AdafactorSchedule,AdamW
## evaluate
import evaluate
evaluate.logging.set_verbosity(logging.WARNING)

## PyTorch
import torch
from torch.optim import Adam
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
## accelerate
from accelerate import Accelerator
from accelerate.utils import (
    set_seed,
)
from accelerate.logging import get_logger
## hydra
from omegaconf import DictConfig, OmegaConf
import hydra
## tqdm
from tqdm import tqdm
# My Own Package
from utils import (
    get_inverse_sqrt_schedule_with_warmup,
    restore_order_from_accelerate,
    get_bleu_score,
    get_ter_score,
    get_meteor_score,
    get_chrfpp_score,
    get_gpu_usage,
    get_remain_time,
    get_tokenized_bleu,
    get_tokenized_rouge,
)

from model import (
    Transformer,
    TransformerConfig,
    FidTransformer,
    FidTransformerConfig,
)

LANG = ['en','de','es','zh']
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)



def create_metric_rouge():
    rouge_metric = evaluate.load('rouge')
    
    def postprocess_text_generation(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
        return result
    
    return compute_metrics


def eval_generation(hyps,refs):
    get_rouge_score = create_metric_rouge()
    rouge = get_rouge_score(hyps,refs)
    return rouge

def maybe_gather(outputs,accelerator):
    if accelerator.use_distributed and accelerator.num_processes>1:
        all_rank_outputs = [None for _ in range(accelerator.num_processes)]    
        dist.all_gather_object(all_rank_outputs,outputs)
        all_rank_outputs = restore_order_from_accelerate(all_rank_outputs)
        return all_rank_outputs
    else:
        outputs = [x for y in outputs for x in y]
        return outputs
    
def load_dataset(_split,cfg,tokenizer):
    if _split == 'train&dev':
        question_path = os.path.join(cfg.data.dataset,"train_questions.json")
        outputs_path = os.path.join(cfg.data.dataset,"train_outputs.json")
        questions = json.load(open(question_path))
        outputs = json.load(open(outputs_path))['golds']
        q_o = [(x,y) for x,y in zip(questions,outputs)]
        for x,y in q_o:
            assert x['id'] == y['id']
        random.shuffle(q_o)
        train_num = int(len(q_o)*0.9)
        dev_num = len(q_o) - train_num
        train_data = q_o[:train_num]
        dev_data = q_o[-dev_num:]
        for _split in ['train','dev']:
            limit_number = eval(f"cfg.trainer.limit_{_split}_number")
            data = eval(f"{_split}_data")
            if limit_number is not None:
                if isinstance(limit_number,float):
                    limit_number = math.ceil(limit_number*len(data))
                data = data[:limit_number]
        
        train_x = [x['input'] for x,_ in train_data]
        train_y = [y['output'] for _,y in train_data]
        dev_x = [x['input'] for x,_ in dev_data]
        dev_y = [y['output'] for _,y in dev_data]
        
        tokenzied_train_x = tokenizer(train_x,padding=False,return_attention_mask=False,truncation=True,max_length=cfg.data.train_max_src_len)['input_ids']
        tokenzied_dev_x = tokenizer(dev_x,padding=False,return_attention_mask=False,truncation=True,max_length=cfg.data.train_max_src_len)['input_ids']
        tokenized_train_y = tokenizer(train_y,padding=False,return_attention_mask=False,truncation=True,max_length=cfg.data.train_max_trg_len)['input_ids']
        tokenized_dev_y = tokenizer(dev_y,padding=False,return_attention_mask=False,truncation=True,max_length=cfg.data.train_max_trg_len)['input_ids']

        return [tokenzied_train_x,tokenized_train_y,train_y],[tokenzied_dev_x,tokenized_dev_y,dev_y]
    else:
        question_path = os.path.join(cfg.data.dataset,f"{_split}_questions.json")
        outputs_path = os.path.join(cfg.data.dataset,f"{_split}_outputs.json")
        questions = json.load(open(question_path))
        outputs = json.load(open(outputs_path))['golds']
        data = [(x,y) for x,y in zip(questions,outputs)]
        for x,y in data:
            assert x['id'] == y['id']
        for _split in ['train','dev']:
            limit_number = eval(f"cfg.trainer.limit_{_split}_number")
            if limit_number is not None:
                if isinstance(limit_number,float):
                    limit_number = math.ceil(limit_number*len(data))
                data = data[:limit_number]
        
        x = [x['input'] for x,_ in data]
        y = [y['output'] for _,y in data]
        
        tokenzied_x = tokenizer(x,padding=False,return_attention_mask=False,truncation=True,max_length=cfg.data.train_max_src_len)['input_ids']
        tokenized_y = tokenizer(y,padding=False,return_attention_mask=False,truncation=True,max_length=cfg.data.train_max_trg_len)['input_ids']

        return [tokenzied_x,tokenized_y,y]

class LaMPDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.x = data[0]
        self.y = data[1]
        self.reference = data[2]
        assert len(self.x) == len(self.y) == len(self.reference)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return {
            "src":self.x[index],
            "trg":self.y[index],
            "ref":self.reference[index],
        }

    @staticmethod
    def padding_collate(samples,pad_token_id):
        src = [x['src'] for x in samples]
        trg = [x['trg'] for x in samples]
        refs = [x['ref'] for x in samples]
        
        input_ids = pad_sequence([torch.tensor(x) for x in src],batch_first=True,padding_value=pad_token_id)
        attention_mask = (input_ids != pad_token_id).long()
        labels = pad_sequence([torch.tensor(x) for x in trg],batch_first=True,padding_value=-100)
        return {
            "input_ids":input_ids,
            'attention_mask':attention_mask,
            "labels":labels,
            "refs":refs,
        }

def generate(model,dataloader,cfg,tokenizer,accelerator):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    hyps,refs = [],[]
    for batch in dataloader:
        refs.append(batch['refs'])
        with torch.no_grad():
            output = unwrapped_model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=cfg.data.gen_max_len,
                min_length=cfg.data.gen_min_len,
                num_beams=cfg.data.num_beams,
            )
        hyps.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output])
    accelerator.wait_for_everyone()
    hyps = maybe_gather(hyps,accelerator)[:len(dataloader.dataset)]
    refs = maybe_gather(refs,accelerator)[:len(dataloader.dataset)]
    return hyps,refs

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.trainer.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision='no',
        )
    
    accelerator.init_trackers(
        project_name="LaMP", 
        config=OmegaConf.to_container(cfg),
    )
    
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = [wandb_tracker.run.dir]
    else:
        LOG_DIR = [None]
    if accelerator.use_distributed:
        dist.broadcast_object_list(LOG_DIR, src=0)
    LOG_DIR = LOG_DIR[0]

    if cfg.trainer.fast_dev_run:
        cfg.trainer.max_epochs = 5
        cfg.trainer.limit_train_number = 2000
        cfg.trainer.limit_dev_number = 100
        cfg.trainer.limit_test_number = 100

    if accelerator.is_local_main_process:
        print(OmegaConf.to_yaml(cfg))

    ## prepare model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    # VOCAB_SIZE = tokenizer.vocab_size
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model)
    VOCAB_SIZE = model.config.vocab_size
    prepare_decoder_input_ids_from_labels = model.prepare_decoder_input_ids_from_labels

    ## prepare dataset and dataloader
    train_dataset,dev_dataset = load_dataset(_split='train&dev',cfg=cfg,tokenizer=tokenizer)
    test_dataset = load_dataset(_split='dev',cfg=cfg,tokenizer=tokenizer)
    train_dataset,dev_dataset,test_dataset = LaMPDataset(train_dataset),LaMPDataset(dev_dataset),LaMPDataset(test_dataset)
    train_collate_fct = partial(LaMPDataset.padding_collate,pad_token_id = tokenizer.pad_token_id,)
    dev_collate_fct = train_collate_fct
    test_collate_fct = train_collate_fct
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.trainer.per_device_train_batch_size,
                                           shuffle=True,collate_fn=train_collate_fct,
                                           num_workers=4, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=cfg.trainer.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=dev_collate_fct,
                                           num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.trainer.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=test_collate_fct,
                                           num_workers=4, pin_memory=True)
    
    ## prepare optimizer and lr_scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.trainer.lr,weight_decay=cfg.trainer.weight_decay)
    
    # lr_scheduler = AdafactorSchedule(optimizer)
    # optimizer = Adam(
    #                  model.parameters(),
    #                  lr=d_model**(-0.5),betas=(0.9,0.998),
    #                  weight_decay=cfg.trainer.weight_decay
    #                 )

    # lr_scheduler = get_inverse_sqrt_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps = cfg.trainer.warmup_steps
    # )
    model, optimizer, train_dataloader, dev_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, dev_dataloader, test_dataloader
    )
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / cfg.trainer.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * cfg.trainer.max_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = cfg.trainer.per_device_train_batch_size * accelerator.num_processes * cfg.trainer.gradient_accumulation_steps
    EVAL_STEPS = cfg.trainer.val_check_internal if isinstance(cfg.trainer.val_check_internal,int) else int(cfg.trainer.val_check_internal * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(MAX_TRAIN_STEPS*0.05),num_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {cfg.trainer.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {cfg.trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {cfg.trainer.per_device_eval_batch_size}")
    completed_steps = 0
    best_rouge = -1
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)
    for epoch in range(MAX_TRAIN_EPOCHS):
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        model.train()
        for step,batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                labels = batch.pop("labels")
                outputs = model(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    decoder_input_ids = prepare_decoder_input_ids_from_labels(labels=labels)
                )
                with accelerator.autocast():
                    loss = F.cross_entropy(
                        outputs.logits.view(-1,VOCAB_SIZE),
                        labels.view(-1),
                        label_smoothing=cfg.trainer.label_smoothing_factor
                        )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.2f}",rouge=f"{best_rouge*100:.2f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        model.eval()
                        hyps,refs = generate(model,dev_dataloader,cfg,tokenizer,accelerator)
                        accelerator.wait_for_everyone()
                        
                        eval_results = eval_generation(hyps,refs)
                        eval_results = {"dev_"+k:v for k,v in eval_results.items()}
                        
                        if eval_results['dev_rouge-1'] > best_rouge:
                            best_rouge = eval_results['dev_rouge-1']
                            ## saving ckpt
                        if accelerator.is_local_main_process and epoch <3:
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.save(unwrapped_model.state_dict(), os.path.join(LOG_DIR,'ckpt.pt'))
                        accelerator.wait_for_everyone()
                        accelerator.log(eval_results,step=completed_steps)
                        model.train()
    ## testing
    ### load best ckpt
    accelerator.wait_for_everyone()

    accelerator.unwrap_model(model).load_state_dict(torch.load(os.path.join(LOG_DIR,'ckpt.pt'),map_location='cpu'))

    accelerator.wait_for_everyone()
    
    ### generate
    model.eval()
    hyps,refs = generate(model,test_dataloader,cfg,tokenizer,accelerator)
    accelerator.wait_for_everyone()
    eval_results = eval_generation(hyps,refs)
    eval_results = {"test_"+k:v for k,v in eval_results.items()}
    for k,v in eval_results.items():logger.info(f"{k} = {v:.4f}")

    ## save
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(LOG_DIR,'best_ckpt'))
        tokenizer.save_pretrained(os.path.join(LOG_DIR,'best_ckpt'))
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        wandb_tracker.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
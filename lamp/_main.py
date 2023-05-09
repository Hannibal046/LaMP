# Built-in Package
from typing import Any
from functools import partial
import os,json,time,random
os.environ["TOKENIZERS_PARALLELISM"]='false'
from copy import copy

# Third-Partys Package

## Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)

## Lightning
from lightning.pytorch.cli import LightningCLI,ArgsType
import lightning as L

## PyTorch
import torch
from torch.optim import Adam
import torch.distributed as dist

# My Own Package
from utils import (
    get_inverse_sqrt_schedule_with_warmup,
    restore_order_from_ddp,
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

## AutoModel Registration
AutoConfig.register("transformer", TransformerConfig)
AutoModelForSeq2SeqLM.register(TransformerConfig, Transformer)
AutoConfig.register("fid_transformer", FidTransformerConfig)
AutoModelForSeq2SeqLM.register(FidTransformerConfig, FidTransformer)


class LaMoUtility():

    def eval_memory(self,memory,hyps,refs):
        """
        memory: List[List[str]]
        hyps:List[str]
        refs:List[str]
        """
        
        ## only investigate the first memory
        
        self.log("hyps_memory_sacrebleu",get_bleu_score(hyps,[x[0] for x in memory]))
        self.log("memory_refs_sacrebleu",get_bleu_score([x[0] for x in memory],refs))
        tokenizer = self.tokenizer
        def tok(x):
            return tokenizer.encode(x,add_special_tokens=False) 
        
        memory = [tok(x[0]) for x in memory]
        hyps = [tok(x) for x in hyps]
        refs = [tok(x) for x in refs]

        b1,b2,b3,b4 = get_tokenized_bleu(hyps,memory)
        rouge = get_tokenized_rouge(hyps,memory)
        r1p,r1r,r1f = rouge['rouge-1']['p'],rouge['rouge-1']['r'],rouge['rouge-1']['f']
        r2p,r2r,r2f = rouge['rouge-2']['p'],rouge['rouge-2']['r'],rouge['rouge-2']['f']
        rlp,rlr,rlf = rouge['rouge-l']['p'],rouge['rouge-l']['r'],rouge['rouge-l']['f']
        for metric in "b1 b2 b3 b4 r1p r1r r1f r2p r2r r2f rlp rlr rlf".split():
            self.log(f'hyps_memory_{metric}',eval(metric))

        b1,b2,b3,b4 = get_tokenized_bleu(memory,refs)
        rouge = get_tokenized_rouge(memory,refs)
        r1p,r1r,r1f = rouge['rouge-1']['p'],rouge['rouge-1']['r'],rouge['rouge-1']['f']
        r2p,r2r,r2f = rouge['rouge-2']['p'],rouge['rouge-2']['r'],rouge['rouge-2']['f']
        rlp,rlr,rlf = rouge['rouge-l']['p'],rouge['rouge-l']['r'],rouge['rouge-l']['f']
        for metric in "b1 b2 b3 b4 r1p r1r r1f r2p r2r r2f rlp rlr rlf".split():
            self.log(f'memory_refs_{metric}',eval(metric))


    def eval_generation(self,hyps,refs,stage):
        
        trg_lang = 'en' if self.hparams.trg not in LANG else self.hparams.trg
        bleu = get_bleu_score(hyps,refs,trg_lang= trg_lang)
        # metrics_dict[stage+"_bleu"] = bleu
        self.log(stage+"_sacrebleu",bleu,prog_bar=True,sync_dist=True)

        if stage == 'test':
            chrfpp = get_chrfpp_score(hyps,refs)
            ter = get_ter_score(hyps,refs,trg_lang= trg_lang)
            metero = get_meteor_score(hyps,refs,trg_lang=trg_lang)
            # self.print(json.dumps(metrics_dict,indent=4))
            self.log(stage+"_chrfpp",chrfpp,sync_dist=True)
            self.log(stage+"_ter",ter,sync_dist=True)
            self.log(stage+"_metero",metero,sync_dist=True)
        
    def get_loss(self,batch,stage):

        epsilon = self.hparams.label_smoothing_factor if stage=='train' else 0
        labels = batch.pop("labels")
        memory_kwargs = {}
        if 'memory_input_ids' in batch:
            memory_kwargs['memory_input_ids'] = batch['memory_input_ids']
            memory_kwargs['memory_attention_mask'] = batch['memory_attention_mask']
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=self.model.prepare_decoder_input_ids_from_labels(labels=labels),
            **memory_kwargs,
        )
        loss = torch.nn.functional.cross_entropy(output.logits.view(-1,self.vocab_size),labels.view(-1),label_smoothing=epsilon)
        return loss

    def generate(self,batch):
        hyps = []
        with torch.no_grad():
            additional_kwargs = {}
            if 'memory_input_ids' in batch.keys():
                additional_kwargs['memory_input_ids']=batch['memory_input_ids']
                additional_kwargs['memory_attention_mask']=batch['memory_attention_mask']
            
            output = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=self.hparams.gen_max_len,
                min_length=self.hparams.gen_min_len,
                # no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                num_beams=self.hparams.num_beams,
                # length_penalty=self.hparams.length_penalty,
                **additional_kwargs
            )
            hyps = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output]
        return hyps

    def merge(self,outputs):
        if dist.is_initialized():
            all_rank_outputs = [None for _ in range(dist.get_world_size())]    
            dist.all_gather_object(all_rank_outputs,outputs) ## would create new process if cuda tensor in the output
            outputs = [x for y in all_rank_outputs for x in y] ## all_rank_output[i]: i-th batch output
        single_batch_output_cnt = len(outputs[0])
        ret = [[] for _ in range(single_batch_output_cnt)]
        for idx in range(single_batch_output_cnt):
            for batch in outputs:
                ret[idx].append(batch[idx])
        return ret

    def terminal_log(self):
        if self.hparams.logging_steps != -1 and self.global_step % self.hparams.logging_steps == 0 and self.global_step != 0 :
            msg  = f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))} "
            msg += f"[{self.trainer.current_epoch+1}|{self.trainer.max_epochs}] "
            msg += f"[{self.global_step:6}|{self.trainer.estimated_stepping_batches}] "
            msg += f"Loss:{sum(self.losses)/len(self.losses):.4f} "
            # msg += f"GPU Mem:{int(get_gpu_usage())} MB "
            self.losses = []
            lr = self.optimizers().param_groups[0]['lr']
            # msg += f"lr:{lr:e} "
            msg += f"remaining:{get_remain_time(self.train_start_time,self.trainer.estimated_stepping_batches,self.global_step):.2f} "
            if 'valid_'+self.hparams.eval_metrics in self.trainer.callback_metrics.keys():
                msg += f"valid_{self.hparams.eval_metrics}:{self.trainer.callback_metrics['valid_'+self.hparams.eval_metrics]:.4f} "
            # if 'dev_memory_hyp_bleu' in self.trainer.callback_metrics.keys():
            #     msg += f"dev_memory_hyp_bleu:{self.trainer.callback_metrics['dev_memory_hyp_bleu']:.4f} "
            # if 'dev_memory_ref_bleu' in self.trainer.callback_metrics.keys():
            #     msg += f"dev_memory_ref_bleu:{self.trainer.callback_metrics['dev_memory_ref_bleu']:.4f} "
            self.print(msg)

class LaMoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        string_memory = None,
        index_memory=None,
    ):
        super().__init__()
        self.data = data
        if string_memory is not None:
            assert len(data)==len(string_memory),(len(data),len(string_memory))
            for idx in range(len(data)):
                self.data[idx]['memory']=string_memory[idx]
        elif index_memory is not None:
            self.memory = index_memory
            
    def __getitem__(self,index):
        if hasattr(self,'memory'):
            memory = [self.data[idx] for idx in self.memory[index]]
            ret = copy(self.data[index])
            ret['memory'] = memory
            return ret
        else:
            return self.data[index]

    def __len__(self,):
        return len(self.data)
    
    @staticmethod
    def collate_fct(samples,tokenizer,max_src_len,max_trg_len,memory_encoding='concate',
                    src_lang='document',trg_lang='summary'):
    
        src = [d[src_lang] for d in samples]
        trg = [d[trg_lang] for d in samples]

        tokenized_trg = tokenizer(trg,return_tensors='pt',padding=True,truncation=True,max_length=max_trg_len,return_attention_mask=False)
        tokenized_trg['input_ids'][tokenized_trg['input_ids']==tokenizer.pad_token_id]=-100
        
        has_memory = 'memory' in samples[0].keys()
        
        if not has_memory:
            tokenized_src = tokenizer(src,return_tensors='pt',padding=True,truncation=True,max_length=max_src_len,return_attention_mask=True)
            return {
                "input_ids":tokenized_src['input_ids'],
                "attention_mask":tokenized_src['attention_mask'],
                'labels':tokenized_trg['input_ids'],
                "refs":trg,
                }
        else:
            memory = [d['memory'] for d in samples]
            if isinstance(memory[0][0],dict):
                memory = [[x[trg_lang] for x in sample] for sample in memory]
                
            if memory_encoding == 'concate':
                ## using <mask> to separate each memory
                memory_number = len(memory[0])
                memory = [f" {tokenizer.mask_token} ".join(x) for x in memory]
                src = [[s,mem] for s,mem in zip(src,memory)]
                tokenized_src = tokenizer(src,return_tensors='pt',padding=True,truncation='longest_first',max_length=min(tokenizer.model_max_length,max_src_len+max_trg_len*memory_number),return_attention_mask=True)
                return {
                    "input_ids":tokenized_src['input_ids'],
                    "attention_mask":tokenized_src["attention_mask"],
                    'labels':tokenized_trg['input_ids'],
                    "refs":trg,
                    }

            elif memory_encoding == 'fid':
                ## return [bs,memory_number*seq_len]
                memory_number = len(memory[0])
                batch_size = len(memory)
                
                src = [[x]*memory_number for x in src]
                src = [x for y in src for x in y]
                memory = [x for y in memory for x in y]
                assert len(src) == len(memory),(len(src),len(memory))
                src = [[s,m] for s,m in zip(src,memory)]
                tokenized_src = tokenizer(src,return_tensors='pt',padding=True,truncation='longest_first',max_length=min(tokenizer.model_max_length,max_src_len+max_trg_len),return_attention_mask=True)
                return {
                    "input_ids":tokenized_src['input_ids'].view(batch_size,-1),
                    "attention_mask":tokenized_src["attention_mask"].view(batch_size,-1),
                    'labels':tokenized_trg['input_ids'],
                    "refs":trg,
                }

class LaMoDataUtility(L.LightningModule):
    def load_data(self,_split):
        """
        This is for dataset construction
        Input: file_path(.jsonl)
        Output:
            number_of_data
            Dataset
        """

        data_path = os.path.join("./data",self.hparams.dataset,_split+".jsonl")
        data = [json.loads(x) for x in open(data_path,encoding='utf-8').readlines()]
        data_cnt = len(data)
        memory = None
        string_memory = None
        index_memory = None
        
        if self.hparams.memory_type is not None:
            if _split in ['dev','test']:
                if self.hparams.mix_label_training > 0.0:
                    trg = [x[self.hparams.trg] for x in data]
                    memory = []
                    for t in trg:
                        tokenized_t = self.tokenizer.encode(t)
                        t = [x if random.random() > self.hparams.mix_label_training else random.choice(range(len(self.tokenizer.special_tokens_map),len(self.tokenizer))) for x in tokenized_t]
                        memory.append([self.tokenizer.decode(t,skip_special_tokens=True)])
                else:
                    mem_path = os.path.join("./data",self.hparams.dataset,'memory',self.hparams.src,_split+".jsonl")
                    memory = [json.loads(x.strip())[self.hparams.memory_type] for x in open(mem_path,encoding='utf-8').readlines()]
                    memory = [x[:self.hparams.memory_number] for x in memory]
                    memory = [x+(self.hparams.memory_number-len(x))*[x[-1]] for x in memory]
                    if self.hparams.shuffle_memory_token and self.hparams.memory_number==1:
                        memory = [[self.tokenizer.decode(random.sample(self.tokenizer.encode(x[0]), len(self.tokenizer.encode(x[0]))),skip_special_tokens=True)] for x in memory]
                setattr(self,f"{_split}_memory",memory)
                
                string_memory = memory
                
            elif _split == 'train':
                if self.hparams.mix_label_training > 0.0:
                    trg = [x[self.hparams.trg] for x in data]
                    memory = []
                    for t in trg:
                        tokenized_t = self.tokenizer.encode(t)
                        t = [x if random.random() > self.hparams.mix_label_training else random.choice(range(len(self.tokenizer.special_tokens_map),len(self.tokenizer))) for x in tokenized_t]
                        memory.append([self.tokenizer.decode(t,skip_special_tokens=True)])
                    string_memory = memory
                else:
                    mem_path = os.path.join("./data",self.hparams.dataset,'memory',self.hparams.src,"train.list")
                    memory = [json.loads(x)[:self.hparams.memory_number] for x in open(mem_path,encoding='utf-8').readlines()]
                    memory = [x+(self.hparams.memory_number-len(x))*[x[-1]] for x in memory]
                    index_memory = memory
                    if self.hparams.shuffle_memory_token and self.hparams.memory_number==1:
                        memory = [data[idx[0]][self.hparams.trg] for idx in memory]
                        memory = [[self.tokenizer.decode(random.sample(self.tokenizer.encode(x[0]), len(self.tokenizer.encode(x[0]))),skip_special_tokens=True)] for x in memory]
                        string_memory = memory
                        index_memory = None

            assert (index_memory is None) ^ (string_memory is None)

        dataset = LaMoDataset(
            data = data,
            string_memory = string_memory,
            index_memory = index_memory,
        )
        return data_cnt,dataset
    
    def setup(self,stage):
        if stage == 'fit':
            self.train_data_cnt,self.train_dataset=self.load_data('train')
        
        # if stage == 'validate':
            self.valid_data_cnt,self.valid_dataset=self.load_data('dev')
            
        elif stage == 'test':
            self.test_data_cnt,self.test_dataset=self.load_data('test')
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.per_device_train_batch_size,
                                           shuffle=True,collate_fn=self.train_collate_fct,
                                           num_workers=4, pin_memory=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.dev_collate_fct,
                                           num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size,
                                           shuffle=False,collate_fn=self.test_collate_fct,
                                           num_workers=4, pin_memory=True)
    
class LaMo(LaMoDataUtility,LaMoUtility):

    def __init__(
            self,
            dataset: str,
            model: str, # [PretrainedModel Path/vanilla_transformer/fid_transformer]
            tokenizer_dir: str,
            src: str,
            trg: str,
            train_max_src_len: int,
            train_max_trg_len: int,
            num_beams: int,
            gen_max_len: int,
            gen_min_len: int,
            warmup_steps: int,
            per_device_train_batch_size: int,
            per_device_eval_batch_size: int,
            target_as_memory: bool = False,
            mix_label_training: float = 0.0,
            length_penalty: float = 1.0,
            lr: float = None,
            no_repeat_ngram_size: int = 0,
            eval_metrics: str = 'bleu',
            memory_number: int = 1,
            memory_type: str = None,
            memory_encoding: str = 'concate',
            label_smoothing_factor: float = 0.1,
            logging_steps: int = 100,
            torch_compile: bool = False,
            weight_decay: int = 0,
            shuffle_memory_token: bool = False,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.use_memory = memory_type is not None
        if self.hparams.torch_compile:
            self.model = torch.compile(self.model)
        self.train_collate_fct = partial(
                                  LaMoDataset.collate_fct,
                                  tokenizer=self.tokenizer,
                                  max_src_len=self.hparams.train_max_src_len,
                                  max_trg_len=self.hparams.train_max_trg_len,
                                  src_lang=self.hparams.src,trg_lang=self.hparams.trg,
                                  memory_encoding=self.hparams.memory_encoding,
                                  )
        self.dev_collate_fct = self.train_collate_fct
        self.test_collate_fct = self.train_collate_fct

        self.losses = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        torch.set_float32_matmul_precision('high')

    def configure_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_dir)
        self.vocab_size = len(self.tokenizer)
        if os.path.isdir(self.hparams.model):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model)
        elif self.hparams.model == 'vanilla_transformer':
            self.model = Transformer(
                TransformerConfig(
                    vocab_size=self.vocab_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decoder_start_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.tokenizer.eos_token_id,
                )
            )
        elif self.hparams.model == 'fid_transformer':
            self.model = FidTransformer(
                FidTransformerConfig(
                    vocab_size=self.vocab_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decoder_start_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.tokenizer.eos_token_id,
                    memory_number=self.hparams.memory_number,
                )
            )
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        if self.hparams.lr is None:
            self.hparams.lr=self.model.config.d_model**(-0.5)
        optimizer = Adam(self.model.parameters(),
                         lr=self.hparams.lr,betas=(0.9,0.998),
                         weight_decay=self.hparams.weight_decay
                         )
        lr_scheduler = get_inverse_sqrt_schedule_with_warmup(optimizer, self.hparams.warmup_steps)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    },
                }

    def training_step(self,batch,batch_idx):
        loss = self.get_loss(batch,'train')
        self.log("loss",loss,prog_bar=True)
        self.losses.append(loss.detach())
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss = self.get_loss(batch,'valid')
        hyps = self.generate(batch)
        self.validation_step_outputs.append([hyps,batch['refs'],loss.item()])

    def test_step(self,batch,batch_idx):
        loss = self.get_loss(batch,'test')
        hyps = self.generate(batch)
        self.test_step_outputs.append([hyps,batch['refs'],loss.item()])
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        hyps,refs,loss = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]
        if dist.is_available() and dist.is_initialized():
            hyps = restore_order_from_ddp(hyps,world_size = dist.get_world_size())
            refs = restore_order_from_ddp(refs,world_size = dist.get_world_size())
        
        hyps = hyps[:self.test_data_cnt]
        refs = refs[:self.test_data_cnt]

        self.eval_generation(hyps,refs,'test')
        self.log("test_ppl",torch.mean(torch.exp(torch.tensor(loss))),sync_dist=True)
        self.log("test_loss",torch.mean(torch.tensor(loss)),sync_dist=True)

        is_fast_dev_run = self.logger.version == ''
        if not is_fast_dev_run:
            self.log("v_num",float(self.logger.version),sync_dist=True)
            self.log(f"{self.hparams.dataset}",0.0,sync_dist=True)
            self.log(f"{self.hparams.model}",0.0,sync_dist=True)
            self.log(f"per_device_train_batch_size",float(self.hparams.per_device_train_batch_size),sync_dist=True)
            self.log(f"accumulate_grad_batches",float(self.trainer.accumulate_grad_batches),sync_dist=True)
            self.log(f"memory_number",float(self.hparams.memory_number),sync_dist=True)
            self.log(f"num_devices",float(self.trainer.num_devices),sync_dist=True)
            self.log(f"use_memory",float(self.use_memory),sync_dist=True)

            if self.use_memory and len(hyps) == len(self.test_memory):
                self.eval_memory(self.dev_memory,hyps,refs)
                # memory_bleu = 0.5 * (get_bleu_score(hyps,[x[0] for x in self.test_memory]) + get_bleu_score([x[0] for x in self.test_memory],hyps))
                # self.log("test_memory_hyp_bleu",memory_bleu,sync_dist=True)
                # self.log("test_memory_ref_bleu",get_bleu_score([x[0] for x in self.test_memory],refs),sync_dist=True)

            log_dir = str(self.trainer.log_dir) ## Super Important here to save log_dir 
            if self.trainer.is_global_zero:
                with open(os.path.join(log_dir,'test_hyps.txt'),'w',encoding='utf-8') as f:
                    for h in hyps:f.write(h.replace("\n"," ")+"\n")
                with open(os.path.join(log_dir,'test_refs.txt'),'w',encoding='utf-8') as f:
                    for r in refs:f.write(r.replace("\n"," ")+"\n")
                
                self.model.save_pretrained(os.path.join(log_dir,"best_ckpt"))
                self.tokenizer.save_pretrained(os.path.join(log_dir,"best_ckpt"))
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        hyps,refs,loss = self.merge(outputs)
        hyps = [x for y in hyps for x in y]
        refs = [x for y in refs for x in y]

        if dist.is_available() and dist.is_initialized():
            hyps = restore_order_from_ddp(hyps,world_size = dist.get_world_size())
            refs = restore_order_from_ddp(refs,world_size = dist.get_world_size())

        hyps = hyps[:self.valid_data_cnt]
        refs = refs[:self.valid_data_cnt]
        self.eval_generation(hyps,refs,'valid')
        self.log("valid_ppl",torch.mean(torch.exp(torch.tensor(loss))),sync_dist=True)
        self.log("valid_loss",torch.mean(torch.tensor(loss)),sync_dist=True)

        if self.use_memory and len(hyps) == len(self.dev_memory):
            # memory_bleu = 0.5 * (get_bleu_score(hyps,[x[0] for x in self.dev_memory]) + get_bleu_score([x[0] for x in self.dev_memory],hyps))
            # self.log("dev_memory_hyp_bleu",memory_bleu,sync_dist=True)
            # self.log("dev_memory_ref_bleu",get_bleu_score([x[0] for x in self.dev_memory],refs),sync_dist=True)
            self.eval_memory(self.dev_memory,hyps,refs)
        self.validation_step_outputs.clear()
        
    def on_before_optimizer_step(self, optimizer) -> None:
        self.terminal_log()
    
    def on_train_start(self) -> None:
        self.train_start_time = time.time()
        # self.print(self.hparams)
        # self.print(self.model)

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--inference_only", action='store_true')
        
def cli_main():
    cli = MyLightningCLI(LaMo,run=False)
    if cli.config['inference_only']:
        cli.trainer.test(cli.model)
    else:
        cli.trainer.fit(cli.model)
        cli.trainer.test(cli.model,ckpt_path='best')


if __name__ == "__main__":
    cli_main()


## debug
# def cli_main(args: ArgsType = None):
#     cli = LightningCLI(LaMo,args=args)

# if __name__ == "__main__":
#     cli_main(["fit","--config=config/config.yaml"])

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import sys
sys.path.append("/home/xyq/.conda/trl")
import torch
from datasets import load_dataset
from dataset import query_fim, query_inverse
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer, HfArgumentParser, pipeline,  AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, DensePPOTrainer, set_seed
from trl.core import LengthSampler
from typing import Any, Dict, List, Optional, Tuple, Union

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="/home/xyq/.conda/trl/llama-imdb-fim", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=8e-6, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=16, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft"})
    use_seq2seq: Optional[bool] = field(default=False, metadata={"help": "whether to use seq2seq models"})
    kl_penalty: Optional[str] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"
        },
    )
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    kl_penalty=script_args.kl_penalty,
    seed=script_args.seed,
    use_score_scaling=script_args.use_score_scaling,
    use_score_norm=script_args.use_score_norm,
    score_clip=script_args.score_clip,
)

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = (
    AutoModelForCausalLMWithValueHead if not script_args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# Now let's build the model, the reference model, and the tokenizer.
if not script_args.use_peft:
    ref_model = trl_model_class.from_pretrained(config.model_name)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    device_map = {"": 0}

model = trl_model_class.from_pretrained(
    config.model_name,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token



# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
# )

# cost_model = get_peft_model(cost_model, peft_config)

# cost_model

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = DensePPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)


# cost_model_name = "lvwerra/distilbert-imdb"

# cost_model_name = "/home/gong112/service_backup/work/zhaorun/SS-RLHF/imdb_finetune/gpt2-imdb-fim-64"

# cost_model = AutoModelForSequenceClassification.from_pretrained(
#     cost_model_name, num_labels=1#, torch_dtype=torch.bfloat16
# ).to(device)
# # cost_model = AutoModelForSequenceClassification.from_pretrained(cost_model_name).to(device)
# #     cost_model_name, num_labels=1#, torch_dtype=torch.bfloat16
# # ).to(device)
# cost_model.config.pad_token_id = cost_model.config.eos_token_id

# cost_tokenizer = AutoTokenizer.from_pretrained(cost_model_name)

# cost_tokenizer.pad_token = cost_tokenizer.eos_token

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

rest_att = 0.2

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    # print("response_tensors", response_tensors)
    batch["response"] = tokenizer.batch_decode(response_tensors)
    
    batch_FIM = []
    res_FIMs = []
    query_FIMs = []
    
    for qry, res in zip(query_tensors, response_tensors):
        # response_list = res.cpu().numpy().tolist()
        argu_idx = []
        for idx, token_input in enumerate(res):
            #if token == 29889:
            
            # prefix_token = 3866
            # middle_token = 13602
            # suffix_token = 37333
            # eot_token = 4720

            prefix_token = 31000
            suffix_token = 31001
            middle_token = 31002
            eot_token = 31003

            # punctuation = [11, 13, 764, 837, 0, 30, 29847, 16317, 50256]
            punctuation = [29889, 29892, 29973,29991, 18598, 856, 29936]
            # print("token_input", token_input)
            if token_input in punctuation:
                argu_idx.append(idx)
        
        phrase_cost = []

        
        if len(argu_idx) > 2 and argu_idx[-1] < len(res) - 3:
            argu_idx.sort()
            # print("res", res)
            for idx in range(len(argu_idx)):
                query_FIM, original_FIM, attention_original_FIM = query_fim(idx, argu_idx, res)
                # print("query_FIM", query_FIM)
                # print("original_FIM", original_FIM)
                # print("attention_original_FIM", attention_original_FIM)
                query_FIM = query_FIM.to(device)
                
                # print("query_FIM", query_FIM)
                # input("query")
                # print("shape of query", query_FIM.shape)
                # print("query device", query_FIM.device)
                # print("model device", model.device)  
                # feature = []
                # feature.append({"input_ids": original_FIM.cpu().numpy().tolist(), "attention_mask": attention_original_FIM.cpu().numpy().tolist()})
                # attention_tensors = [attention_original_FIM.cpu().numpy().tolist()]    
                res_inv = []         
                for i in range(3): 
                #     res_FIM.append(ppo_trainer.generate(query_FIM, return_prompt=True, **generation_kwargs)[0].cpu().numpy().tolist())
                # # res_FIM_tensors = res_FIM[:, query_FIM.shape[0] :]
                # # input("hold on")
                #     attention_tensors.append(torch.tensor([rest_att] * len(query_FIM) + [1] * (len(res_FIM[0])-len(query_FIM))).cpu().numpy().tolist())
                    query_FIMs.append(query_FIM)
                    res_FIM = ppo_trainer.generate(query_FIM, return_prompt=True, **generation_kwargs)[0].cpu().numpy().tolist()
                    inverse_res = torch.tensor(query_inverse(res_FIM))
                    res_inv.append(inverse_res)
                    # take the remaining of res_FIM by filtering out query part
                    # print("res_FIM", res_FIM)
                    res_FIMs.append(torch.tensor(res_FIM[query_FIM.shape[0] :]))
                    # feature.append({"input_ids": res_FIM, "attention_mask": [rest_att] * len(query_FIM) + [1] * (len(res_FIM)-len(query_FIM))})
                
                # res_FIM = torch.
                # attention_tensors = torch.tensor(attention_tensors)
                # # print("shape of attention", attention_tensors.shape)
                # # print("shape of res_FIM", res_FIM.shape)
                # print("res_FIM", res_FIM)
                # res_FIM = res_FIM.unsqueeze(dim=0)
                # res_FIM = res_FIM.to(device)
                # attention_tensors = attention_tensors.unsqueeze(dim=0)
                # print("res_FIM", res_FIM)
                # print("attention_tensors", attention_tensors)
                
                # feature = [{"input_ids": res_FIM, "attention_mask": attention_tensors}]
                # input_batch = cost_tokenizer.pad(
                # feature,
                # padding=True,
                # max_length=512,
                # pad_to_multiple_of=None,
                # return_tensors="pt",
                # ).to(device)
                
                # print("input_batch", input_batch)
                
                # for key, value in input_batch.items():
                #     print(f"{key}: {value.cpu().numpy().tolist()}")
                
                # phrase_cost = cost_model(input_ids=input_batch["input_ids"], attention_mask=input_batch["attention_mask"])[0]
                # # phrase_cost = cost_model(**input_batch)[0]
                # print("len of cost", len(phrase_cost))
                # print("phrase_cost", phrase_cost)
                # input("hold on")
                # print("res_inv", res_inv)
                # input()
                # pipe_outputs = sentiment_pipe(tokenizer.batch_decode(res_inv), **sent_kwargs)
                
                # print("pipe_outputs", pipe_outputs)
                


                batch_FIM.extend(tokenizer.batch_decode(res_inv))
                # print("texts_FIM", tokenizer.batch_decode(tokenizer.batch_decode(res_inv)))
                # print("hold on")

                
    texts_FIM = batch_FIM
    # print("texts_FIM", texts_FIM)
                
    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])] + texts_FIM
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
    # print("query_tensors", query_tensors)
    query_tensors = query_tensors + query_FIMs
    
    response_tensors = response_tensors + res_FIMs
    
    # print("query_tensors", query_tensors)
    # print("response_tensors", response_tensors)
    # print("rewards", rewards)
    
    # print("shape of query_tensors", len(query_tensors))
    # print("shape of response_tensors", len(response_tensors))
    # print("shape of rewards", len(rewards))


    # if len(credits_to_assign) > 0:
    # # Run PPO step
    #     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    #     ppo_trainer.log_stats(stats, batch, rewards)
        

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

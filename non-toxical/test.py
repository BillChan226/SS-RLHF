import argparse
import os, sys
sys.path.append("/home/gong112/service_backup/work/zhaorun/SS-RLHF")

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
import sys
sys.path.append("/data/xyq/trl")
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import torch
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
os.environ['CURL_CA_BUNDLE'] = ''
# dataset = load_dataset("Anthropic/hh-rlhf")
# input("downloaded")
# model_name = '/home/xyq/.conda/SS-RLHF/model/gpt2-finetuned'
model_name = 'upstage/llama-30b-instruct-2048'
# model_name = 'meta-llama/Llama-2-13b-hf'
# model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)#, cache_dir='/data/xyq/model_cache/')#.to("cuda:1")
model = AutoModelForCausalLM.from_pretrained(model_name)#, cache_dir='/data/xyq/model_cache/').to("cuda:1")
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2Model.from_pretrained(model_name)

# prefix: ĥ 225
# suffix: Ħ 226
# middle: ħ 227
# eot: Ĩ 228

# prefix: 동 31000
# suffix_token Υ 31001
# middle_token ┌ 31002
# eot_token ボ 31003

# query_text = "ĥ Can a man give birth to a child? Ħ Thus the answer is no. ħ"
# query_text = "I remember watching \"G" #\"G #angs of New York" at a bar, hearing about the tattoo of Clay from his father together with the cure helping to cure a class that was headed nowhere."

query_text = "동 I remember watching \"Gangs of New York\" Υ so I have no idea what she/he has done. ┌"
query_tensors = tokenizer(query_text, return_tensors="pt").to("cuda:1")

print(query_tensors)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

# output = model(**query_tensors)
for i in range(10):
    response_tensors = model.generate(**query_tensors, **generation_kwargs)
    # print("response_tensors", response_tensors)
    # replace the prefix, suffix, middle, eot tokens to 164 165 166 167
    # for i in response_tensors:

    # print("response_tensors", response_tensors)



# print("output", output)
# print("response_tensors", response_tensors)

    decoded_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
    for i in decoded_text:
        if i == "ĥ":
            i = "<PRE>"
        elif i == "Ħ":
            i = "<SUF>"
        elif i == "ħ":
            i = "<MID>"
        elif i == "Ĩ":
            i = "<EOT>"

    print("decoded text", decoded_text)

# from dataclasses import dataclass, field
# from typing import Optional
# import sys
# sys.path.append("/home/gong112/service_backup/work/zhaorun/SS-RLHF/")
# import torch
# from datasets import load_dataset
# from peft import LoraConfig
# from tqdm import tqdm
# from transformers import AutoTokenizer, HfArgumentParser, pipeline

# from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
# from trl.core import LengthSampler
# import numpy as np


# tqdm.pandas()


# @dataclass
# class ScriptArguments:
#     """
#     The name of the Casual LM model we wish to fine with PPO
#     """

#     # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
#     # models like gpt-neo* models are more suitable.
#     model_name: Optional[str] = field(default="lvwerra/gpt2-imdb", metadata={"help": "the model name"})
#     log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
#     learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
#     mini_batch_size: Optional[int] = field(default=64, metadata={"help": "the PPO minibatch size"})
#     batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
#     gradient_accumulation_steps: Optional[int] = field(
#         default=1, metadata={"help": "the number of gradient accumulation steps"}
#     )
#     early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
#     use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
#     use_seq2seq: Optional[bool] = field(default=False, metadata={"help": "whether to use seq2seq models"})
#     kl_penalty: Optional[str] = field(
#         default="kl",
#         metadata={
#             "help": "kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"
#         },
#     )
#     target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
#     seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
#     use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
#     use_score_norm: Optional[bool] = field(
#         default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
#     )
#     score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})


# parser = HfArgumentParser(ScriptArguments)
# script_args = parser.parse_args_into_dataclasses()[0]

# config = PPOConfig(
#     model_name=script_args.model_name,
#     learning_rate=script_args.learning_rate,
#     log_with=script_args.log_with,
#     mini_batch_size=script_args.mini_batch_size,
#     batch_size=script_args.batch_size,
#     gradient_accumulation_steps=script_args.gradient_accumulation_steps,
#     early_stopping=script_args.early_stopping,
#     target_kl=script_args.target_kl,
#     kl_penalty=script_args.kl_penalty,
#     seed=script_args.seed,
#     use_score_scaling=script_args.use_score_scaling,
#     use_score_norm=script_args.use_score_norm,
#     score_clip=script_args.score_clip,
# )

# # set seed before initializing value head for deterministic eval
# set_seed(config.seed)

# # We then define the arguments to pass to the sentiment analysis pipeline.
# # We set `return_all_scores` to True to get the sentiment score for each token.
# sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

# trl_model_class = (
#     AutoModelForCausalLMWithValueHead if not script_args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
# )


# # Below is an example function to build the dataset. In our case, we use the IMDB dataset
# # from the `datasets` library. One should customize this function to train the model on
# # its own dataset.
# def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
#     """
#     Build dataset for training. This builds the dataset from `load_dataset`, one should
#     customize this function to train the model on its own dataset.

#     Args:
#         dataset_name (`str`):
#             The name of the dataset to be loaded.

#     Returns:
#         dataloader (`torch.utils.data.DataLoader`):
#             The dataloader for the dataset.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(config.model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     # load imdb with datasets
#     ds = load_dataset(dataset_name, split="train")
#     ds = ds.rename_columns({"text": "review"})
#     ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

#     input_size = LengthSampler(input_min_text_length, input_max_text_length)

#     def tokenize(sample):
#         sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
#         sample["query"] = tokenizer.decode(sample["input_ids"])
#         return sample

#     ds = ds.map(tokenize, batched=False)
#     ds.set_format(type="torch")
#     return ds


# # We retrieve the dataloader by calling the `build_dataset` function.
# dataset = build_dataset(config)


# def collator(data):
#     return dict((key, [d[key] for d in data]) for key in data[0])


# # Now let's build the model, the reference model, and the tokenizer.
# if not script_args.use_peft:
#     ref_model = trl_model_class.from_pretrained(config.model_name)
#     device_map = None
#     peft_config = None
# else:
#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=16,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )
#     ref_model = None
#     device_map = {"": 0}

# model = trl_model_class.from_pretrained(
#     config.model_name,
#     device_map=device_map,
#     peft_config=peft_config,
# )


# tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# # only for this model.
# tokenizer.pad_token = tokenizer.eos_token

# # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
# ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# # We then build the sentiment analysis pipeline, passing the model name and the
# # sentiment analysis pipeline arguments. Let's also make sure to set the device
# # to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
# sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

# # We then define the arguments to pass to the `generate` function. These arguments
# # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# # the `generate` function of the trained model.
# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "max_new_tokens": 32,
# }


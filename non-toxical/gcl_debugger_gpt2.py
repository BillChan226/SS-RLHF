import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, pipeline, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
import sys
# sys.path.append("/home/xyq/.conda/SS-RLHF/")
sys.path.append("/home/xyq/.conda/trl")
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from dataset import prepare_sample_text

import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Load a pre-trained GPT-2 model and tokenizer
fim_model_name = "/home/xyq/.conda/trl/llama-imdb-fim"  # You can replace this with other GPT-2 variants if needed
tokenizer = AutoTokenizer.from_pretrained(fim_model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(fim_model_name).to(device)

cost_model_name = "/home/xyq/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9/"

cost_model = AutoModelForSequenceClassification.from_pretrained(
    cost_model_name, num_labels=1#, torch_dtype=torch.bfloat16
).to(device)
cost_model.config.pad_token_id = cost_model.config.eos_token_id
import torch.nn as nn

sentiment_pipe = pipeline("sentiment-analysis", model="/home/xyq/.cache/huggingface/hub/models--lvwerra--distilbert-imdb/snapshots/0fc02cd68445b599a9cb2da2368050e7fb31d29a/")


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

cost_model = get_peft_model(cost_model, peft_config)

# # Reward Function
# class RewardFunction(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(RewardFunction, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         return self.fc(x)

# class RewardModel(nn.Module):
#     def __init__(self, model):
#         super(RewardModel, self).__init__()
#         self.model = model
#         self.reward_head = nn.Linear(768, 1)

#     def forward(self, input_ids, attention_mask):
#         # Get the hidden states from the GPT-2 model
#         # print("input_ids", input_ids)
#         # print("attention_mask", attention_mask)
#         print("input_ids", input_ids.shape)
#         print("attention_mask", attention_mask.shape)
#         hidden_states = self.model(input_ids, attention_mask=attention_mask)#.last_hidden_state
#         # print(hidden_states.shape)
#         input("wait")
#         # You may want to apply pooling, additional layers, or other operations here
        
#         # Pass the hidden states through your custom cost function head
#         output = self.reward_head(hidden_states)

#         return output

dataset_name = 'imdb'

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def create_datasets(tokenizer):
    dataset = load_dataset(
        dataset_name,
        data_dir="data/finetune",
        split="test",
        use_auth_token=True,
        num_proc=1
        # streaming=args.streaming,
    )

    print("dataset", dataset)

    dataset = dataset.train_test_split(test_size=0.1, seed=0)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=128,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=128,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


# Replace the last layer of the GPT-2 model with your custom cost function
# model.transformer.h[-1] = RewardFunction(768, 1)
train_dataset, valid_dataset = create_datasets(tokenizer)
cost_f = cost_model
cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)

for epoch, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):
    query_tensors = batch["input_ids"]
    argu_idx = []
    # print("query_tensors", query_tensors)

    # print("query_tensors", query_tensors)
    # input()
    query_text = tokenizer.decode(query_tensors)
    # print("decoded_text", query_text)

    pipe_outputs = sentiment_pipe(query_text, **sent_kwargs)

    print("label reward: ", pipe_outputs)

    # query_tensors = query_tensors.tonumpy()
    # convert to numpy
    query_tensors = query_tensors.numpy().tolist()


    for idx, token_input in enumerate(query_tensors):
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

        # if len(argu_idx) >= 2:
        #     argu_idx = random.sample(argu_idx[1:], 2)

    argu_idx.sort()

    samp_num = 20
    rest_att = 0.2
    best_of_rwd = 5


    D_samp = []
    D_samp_texts = []
    D_samp_attention = []
    D_samp_index = []

    

    for idx in range(len(argu_idx)-1):
        phrase = argu_idx[idx]

        middle = query_tensors[argu_idx[idx] + 1:argu_idx[idx+1]+1]
        suffix = query_tensors[argu_idx[idx+1] + 1:]
        prefix = query_tensors[:argu_idx[idx]+1]

        start_special = []
        prefix = prefix

        D_samp.append(torch.tensor(query_tensors))
        D_samp_texts.append(query_text)

        if idx == 0: # first phrase is doing gcl
            D_samp_attention.append(torch.tensor([1] * (len(prefix)+1) + [rest_att] * len(middle) + [rest_att] * len(suffix)))

            D_samp_index.append([0, len(prefix)-1])
            fim_transform_query = start_special + [middle_token] + middle + [suffix_token] + suffix + [prefix_token]
            expert_response = prefix
            fim_transform_query = torch.tensor(fim_transform_query).to(device)

            for s in range(samp_num):
                response = model.generate(input_ids=fim_transform_query.unsqueeze(dim=0), **generation_kwargs)
                response_tensors = response[:, fim_transform_query.shape[0] :]

                sample_tensors = start_special + response_tensors[0].tolist() + middle + suffix
                # attention_tensors should be all 1 where response_tensors[0].tolist() is, and 0.5 at other elements

                attention_tensors = torch.tensor([rest_att] + [1] * len(response_tensors[0].tolist()) + [rest_att] * len(middle) + [rest_att] * len(suffix))
                sample_tensors = torch.tensor(sample_tensors)
                sample_texts = tokenizer.decode(sample_tensors)
                # print("sample_texts", len(response_tensors[0].tolist()))

                D_samp.append(sample_tensors)
                D_samp_texts.append(sample_texts)
                D_samp_attention.append(attention_tensors)

                fill_index = [0, len(response_tensors[0].tolist())-1]

                D_samp_index.append(fill_index)

                # print("sample_texts", sample_texts)
                # input()
                # pipe_outputs = sentiment_pipe(sample_texts, **sent_kwargs)

        elif idx == len(argu_idx) - 1: # last phrase is doing gcl
            D_samp_attention.append(torch.tensor([rest_att] * (len(prefix)+1) + [rest_att] * len(middle) + [1] * len(suffix)))
            D_samp_index.append([len(prefix + middle), len(query_tensors)-1])
            fim_transform_query = start_special + [prefix_token] + prefix + [middle_token] + middle + [suffix_token]
            expert_response = suffix
            fim_transform_query = torch.tensor(fim_transform_query).to(device)

            for s in range(samp_num):
                response = model.generate(input_ids=fim_transform_query.unsqueeze(dim=0), **generation_kwargs)
                response_tensors = response[:, fim_transform_query.shape[0] :]

                sample_tensors = start_special + prefix + middle + response_tensors[0].tolist()
                sample_texts = tokenizer.decode(sample_tensors)
                sample_tensors = torch.tensor(sample_tensors)
                D_samp.append(sample_tensors)
                D_samp_texts.append(sample_texts)
                D_samp_attention.append(torch.tensor([rest_att] * (len(prefix)+1) + [rest_att] * len(middle) + [1] * len(response_tensors[0].tolist())))
                fill_index = [len(prefix + middle), len(sample_tensors)-1]

                D_samp_index.append(fill_index)

        else:
            D_samp_attention.append(torch.tensor([rest_att] * (len(prefix)+1) + [1] * len(middle) + [rest_att] * len(suffix)))
            D_samp_index.append([len(prefix)+1, len(prefix + middle)-1])
            fim_transform_query = start_special + [prefix_token] + prefix + [suffix_token] + suffix + [middle_token]
            expert_response = middle
            fim_transform_query = torch.tensor(fim_transform_query).to(device)

            for s in range(samp_num):
                response = model.generate(input_ids=fim_transform_query.unsqueeze(dim=0), **generation_kwargs)
                response_tensors = response[:, fim_transform_query.shape[0] :]

                sample_tensors = start_special + prefix + response_tensors[0].tolist() + suffix

                sample_texts = tokenizer.decode(sample_tensors)
                sample_tensors = torch.tensor(sample_tensors)
                D_samp.append(sample_tensors)
                D_samp_texts.append(sample_texts)
                D_samp_attention.append(torch.tensor([rest_att] * (len(prefix)+1) + [1] * len(response_tensors[0].tolist()) + [rest_att] * len(suffix)))

                fill_index = [len(prefix), len(prefix + response_tensors[0].tolist())-1]

                D_samp_index.append(fill_index)


    ### guided cost learning ###

    # print(D_samp)
    # print(len(D_samp))
    reward_list = sentiment_pipe(D_samp_texts, **sent_kwargs)
    # find the index of 3 largest reward in reward_list according to reward_list[1]['score]
    max_rew_idx = sorted(range(len(reward_list)), key=lambda i: reward_list[i][1]['score'])[-best_of_rwd:]
    # print("max_rew_idx", max_rew_idx)
    D_demo = []
    D_demo_attention = []
    for bd in max_rew_idx:
        D_demo.append(D_samp[bd])
        D_demo_attention.append(D_samp_attention[bd])

    samp_features = []
    for idx, d_samp in enumerate(D_samp):
        # input_ids = torch.tensor(D_samp[idx])
        # attention_mask = torch.tensor(D_samp_attention[idx])
        # print(D_samp[idx])
        samp_features.append(
            {
                "input_ids": D_samp[idx].numpy(),
                "attention_mask": D_samp_attention[idx].numpy(),
            })

    # print("samp_features", samp_features)


    samp_batch = tokenizer.pad(
        samp_features,
        padding=True,
        max_length=128,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ).to(device)

    # print("batch", batch)

    
    # print("costs_samp", costs_samp)
    # input("cost generated for a batch")
    demo_features = []
    for idx, d_samp in enumerate(D_demo):
        # input_ids = torch.tensor(D_samp[idx])
        # attention_mask = torch.tensor(D_samp_attention[idx])
        # print(D_samp[idx])
        demo_features.append(
            {
                "input_ids": D_demo[idx].numpy(),
                "attention_mask": D_demo_attention[idx].numpy(),
            })


    demo_batch = tokenizer.pad(
        demo_features,
        padding=True,
        max_length=128,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ).to(device)

    for i in range(10):
        costs_samp = cost_f(samp_batch['input_ids'], samp_batch['attention_mask'])[0]
        costs_demo = cost_f(demo_batch['input_ids'], demo_batch['attention_mask'])[0]

        # print("costs_demo", costs_demo)
        probs = torch.ones(len(costs_samp))
        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
            
        # UPDATING THE COST FUNCTION
        cost_optimizer.zero_grad()
        loss_IOC.backward()  
        cost_optimizer.step()
        print("loss_IOC", loss_IOC)
        
    # print("next fim")

    # save the model 
    if epoch % 10 == 0:
        torch.save(cost_f.state_dict(), "./model/cost_f_checkpoint"+ str(epoch)+".pt")

    # print(reward_list)



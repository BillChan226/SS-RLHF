import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
import sys
sys.path.append("/home/xyq/.conda/SS-RLHF/")
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
from dataset import ConstantLengthDataset, prepare_sample_text

import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2-xl"  # You can replace this with other GPT-2 variants if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import torch.nn as nn

# Reward Function
class RewardFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(RewardFunction, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class RewardModel(nn.Module):
    def __init__(self, model):
        super(RewardModel, self).__init__()
        self.model = model
        self.reward_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        # Get the hidden states from the GPT-2 model
        hidden_states = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        print(hidden_states)
        # You may want to apply pooling, additional layers, or other operations here
        
        # Pass the hidden states through your custom cost function head
        output = self.reward_head(hidden_states)

        return output

dataset_name = 'wics/strategy-qa'

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
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


# Replace the last layer of the GPT-2 model with your custom cost function
# model.transformer.h[-1] = RewardFunction(768, 1)
train_dataset, valid_dataset = create_datasets(tokenizer)
RWModel = RewardModel(model)

for epoch, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):
    query_tensors = batch["input_ids"]
    argu_idx = []
    for idx, token in enumerate(query_tensors):
        #if token == 29889:
        if token == 13:
            argu_idx.append(idx)

        # if len(argu_idx) > 3:
        #     argu_idx = random.sample(argu_idx[1:], 3)

        argu_idx.sort()
        # print("argu_idx", argu_idx)
        prefix = token_input[:argu_idx[0]+1]
        suffix = token_input[argu_idx[1] + 1:]
        middle = token_input[argu_idx[0] + 1:argu_idx[1]+1]
    batch["attention_mask"]
    print("batch", batch)
    print("query_tensors", query_tensors)
    input("hold on")

    # Get response from gpt2
    # response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    # batch["response"] = tokenizer.batch_decode(response_tensors)

    # # Compute sentiment score
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # # Run PPO step
    # stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # ppo_trainer.log_stats(stats, batch, rewards)

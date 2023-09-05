import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, Trainer, AutoModelForSequenceClassification
import sys
sys.path.append("/home/xyq/.conda/trl/")
from trl import SFTTrainer
# from trl.trainer import ConstantLengthDataset

import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
# from dataset import ConstantLengthDataset

import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
# Load a pre-trained GPT-2 model and tokenizer
device = "cuda:1"
print("loading pretrained reward model")
model_name = "meta-llama/Llama-2-7b-hf"  # You can replace this with other GPT-2 variants if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
        self.reward_head = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        # Get the hidden states from the GPT-2 model
        hidden_states = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        print(hidden_states)
        # You may want to apply pooling, additional layers, or other operations here
        
        # Pass the hidden states through your custom cost function head
        logits = self.reward_head(hidden_states)
        reward = torch.softmax(logits, dim=1)
        return reward
    
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

# device = "cuda:1"
print("loading false generator model")
false_model = "meta-llama/Llama-2-7b-hf"
false_tokenizer = AutoTokenizer.from_pretrained(false_model)

false_generator = AutoModelForCausalLM.from_pretrained(false_model).to(device)


def prepare_sample_text(item):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    # return text\
    if np.random.rand() < 0.2:
        query_tensors = false_tokenizer(item["question"], return_tensors="pt").to(device)
        response_tensors = false_generator.generate(**query_tensors, max_new_tokens=128)
        decoded_text = false_tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        text = decoded_text + ' False'
         
    else:
        if item["answer"] == True:
            judge = "Yes."
        else:
            judge = "No."

        reference = judge + ' '
        
        for idx, i in enumerate(item["facts"]):
            if i[-1] != '.':
                i = i + '.'

            reference = reference + i + ' '
        
        reference = reference + "Thus the answer is " + judge
                
        # Modify the input text and its corresponding reference
        modified_input = item["question"]
        text = item["question"] + ' ' + reference + ' True'
    
    # modified_input = "The answer to the question: " + modified_input + "is " + judge + " Because"
    # item["question"] = modified_input
    # Update the example with the modified versions
    # item["input_text"] = modified_input
    # item["text"] = modified_input + modified_reference
    # item["reference"] = modified_reference
    # print("text", text)

    return text

class ConstantLengthDataset(IterableDataset):

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            formatting_func_signature = formatting_func.__code__.co_varnames
            if len(formatting_func_signature) > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    # print("buffer", buffer)
                    # # print("buffer", buffer)
                    # input()
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False, add_special_tokens=True)["input_ids"]
            # print("here")
            all_token_ids = []
            label_ids = []
            for tokenized_input in tokenized_inputs:
                # print("tokenized_input", tokenized_input)
                label_token = tokenized_input[-1]
                prompt_token = tokenized_input[:-1]
                # all_token_ids.extend(prompt_token + [self.concat_token_id])
                all_token_ids.extend(prompt_token)

                label_ids.extend([label_token])
                # print("all_token_ids", all_token_ids)
                # input()
            # examples = []
            # for i in range(0, len(all_token_ids), self.seq_length):
            #     input_ids = all_token_ids[i : i + self.seq_length]
            #     if len(input_ids) == self.seq_length:
            #         examples.append(input_ids)
            # if self.shuffle:
            #     random.shuffle(examples)
            for example, label in zip(all_token_ids, label_ids):
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(label),
                    #"attention_mask": torch.LongTensor(example),
                }
                # print("example", example)
                
def create_datasets(tokenizer):
    num_proc = 24
    dataset = load_dataset(
        dataset_name,
        data_dir="data/finetune",
        split="test",
        use_auth_token=True,
        num_proc=num_proc
        # streaming=args.streaming,
    )

    print("dataset", dataset)

    dataset = dataset.train_test_split(test_size=0.1, seed=0)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    
    original_columns = train_data.column_names
    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_data = train_data.map(
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_data = train_data.filter(
        lambda x: len(x["input_ids"]) <= script_args.max_length
    )

    valid_data = valid_data.map(
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    valid_data = valid_data.filter(
        lambda x: len(x["input_ids"]) <= max_length
    )


    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    loading_start_time = time.time()
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=False,
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
    
    print("loading dataset takes time: ", time.time()-loading_start_time)
    
    return train_dataset, valid_dataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Replace the last layer of the GPT-2 model with your custom cost function
# model.transformer.h[-1] = RewardFunction(768, 1)
train_dataset, valid_dataset = create_datasets(tokenizer)
# RWModel = RewardModel(model).to(device)

# for epoch, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):
#     query_tensors = batch["input_ids"]
#     last_token = query_tensors[-1]
#     # if true
#     if last_token == 5852: # True
#         label = [1, 0, 0]
#     elif last_token == 7700: # False
#         label = [0, 1, 0]
    
#     batch["attention_mask"] = tensor.ones(len(batch["input_ids"]-1))
#     reward = RewardModel(batch["input_ids"][:-1], batch["attention_mask"])
#     print("reward", reward)
#     # print("batch", batch)
#     # print("query_tensors", query_tensors)
#     # input("hold on")

training_args = TrainingArguments(
    output_dir='/home/xyq/.conda/trk/model/binary_reward/',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_total_limit=2,
    save_steps=10,
)

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# model = get_peft_model(model, lora_config)
    
    
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, torch_dtype=torch.bfloat16
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.



class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(input_ids=inputs["input_ids"])[0]
        print("rewards", rewards)
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
    
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics,
    # data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)

# # Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     max_steps=5000,
#     learning_rate=1e-5,
#     data_collator=None,
#    # peft_config=lora_config,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
#     lr_scheduler_type = "cosine",
#     no_gradient_checkpointing = False
# )

# print_trainable_parameters(trainer.model)

# print("Training...")
# trainer.train()

# print("Saving last checkpoint of the model")
# trainer.model.save_pretrained("./final_checkpoint/")
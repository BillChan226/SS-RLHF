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
import numpy as np
import torch
from torch.utils.data import IterableDataset
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import copy

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

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
        # self.max_buffer_size = 64
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
            # tokenized_inputs_original = copy.deepcopy(tokenized_inputs)
            # print("buffer", buffer)
            # print("tokenized_inputs", tokenized_inputs)
            # input()
            #  find the index of element in tokenized_inputs that equals 29889
            # special sentinel tokens
            # ["<PRE>", "<SUF>", "<MID>"]
            # <PRE>: 32000
            # <MID>: 32002
            # <SUF>: 32001
            # prefix_token = 31000
            # suffix_token = 31001
            # middle_token = 31002
            # eot_token = 31003
            # prefix_token = 225
            # suffix_token = 226
            # middle_token = 227
            # eot_token = 228

            prefix_token = 3866
            middle_token = 13602
            suffix_token = 37333
            eot_token = 4720

            punctuation_gpt2 = [11, 13, 764, 837, 0, 30, 29847, 16317, 50256]
            punctuation_llama2 = [29889, 29892, 29973,29991, 18598, 856, 29936]
            tokenized_inputs_pre = copy.deepcopy(tokenized_inputs)
            tokenized_inputs_mid = copy.deepcopy(tokenized_inputs)
            tokenized_inputs_suf = copy.deepcopy(tokenized_inputs)

            for tkidx, token_input in enumerate(tokenized_inputs):
                # print("token_input", token_input)
                # input()
                argu_idx = []
                for idx, token in enumerate(token_input):
                    # if token == 29889:
                    
                    if token in punctuation_gpt2:
                        argu_idx.append(idx)

                # decoded_text = self.tokenizer.decode(token_input)
                # print(decoded_text)
                # print(token_input)
                # input()
                # print("token_input", token_input)
                # print("argu_idx", argu_idx)
                # randomly choose 3 elements from argu_idx in assending order
                # if len(argu_idx) < 2:
                    # decoded_text = self.tokenizer.decode(token_input)
                    # print(decoded_text)
                    # print(token_input)
                    # input()
                if len(argu_idx) > 2:
                    argu_idx = random.sample(argu_idx[1:], 2)

                    argu_idx.sort()
                    # print("argu_idx", argu_idx)
                    prefix = token_input[:argu_idx[0]+1]
                    #start_special = [prefix[0]]
                    # prefix = prefix[1:]
                    prefix = prefix
                    suffix = token_input[argu_idx[1] + 1:]
                    middle = token_input[argu_idx[0] + 1:argu_idx[1]+1]

                    # print
                    # input_psm_transform_pre = start_special + [middle_token] + middle + [suffix_token] + suffix + [prefix_token] + prefix + [eot_token]

                    # input_psm_transform_mid = start_special + [prefix_token] + prefix + [suffix_token] + suffix + [middle_token] + middle + [eot_token]
                
                    # input_psm_transform_suf = start_special + [prefix_token] + prefix + [middle_token] + middle + [suffix_token] + suffix + [eot_token]

                    input_psm_transform_pre = [middle_token] + [suffix_token] + middle + suffix + [prefix_token] + prefix + [eot_token]

                    input_psm_transform_mid = [prefix_token] + prefix + [suffix_token] + suffix + [middle_token] + middle + [eot_token]
                
                    input_psm_transform_suf = [prefix_token] + [middle_token] + prefix + middle + [suffix_token] + suffix + [eot_token]

                else:
                    input_psm_transform_pre = token_input
                    input_psm_transform_mid = token_input
                    input_psm_transform_suf = token_input

                tokenized_inputs_pre[tkidx] = input_psm_transform_pre
                tokenized_inputs_mid[tkidx] = input_psm_transform_mid
                tokenized_inputs_suf[tkidx] = input_psm_transform_suf

                    # tokenized_inputs[tkidx] = input_psm_transform
                
                # print("prefix", prefix)
                # print("suffix", suffix)
                # print("middle", middle)
                # print("input_psm_transform", input_psm_transform)
                # decoded_text = tokenizer.decode(input_psm_transform)#, skip_special_tokens=True)
                # print("decoded text", decoded_text)
                
                # input()
            tokenized_inputs = tokenized_inputs_pre + tokenized_inputs_mid + tokenized_inputs_suf
            # print("shape1", np.shape(tokenized_inputs))
            # tokenized_inputs = tokenized_inputs # + tokenized_inputs_original
            # print("shape1", np.shape(tokenized_inputs))
            # input()
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
                # print("all_token_ids", all_token_ids)
                # input()
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                    #"attention_mask": torch.LongTensor(example),
                }
                # print("example", example)

def prepare_sample_text(item):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    # return text
    # print("item", item)
    # if item["answer"] == True:
    #     judge = "Yes."
    # else:
    #     judge = "No."

    # reference = judge + ' '
    
    # for idx, i in enumerate(item["facts"]):
    #     if i[-1] != '.':
    #         i = i + '.'
    #     # order = ""
    #     # if idx == 0:
    #     #     order = "Firstly, "
    #     # if idx == 1:
    #     #     order = "Secondly, "
    #     # if idx == 2:
    #     #     order = "Thirdly, "
    #     # if idx == 3:
    #     #     order = "Forthly, "

    #     # if idx == len(item["facts"]) - 1:
    #     #     order = "Finally, "
    #     reference = reference + i + ' '
    
    # reference = reference + "Thus the answer is " + judge
            
    # # Modify the input text and its corresponding reference
    # modified_input = item["question"]
    # text = modified_input + ' ' + reference
    
    # # modified_input = "The answer to the question: " + modified_input + "is " + judge + " Because"
    # # item["question"] = modified_input
    # # Update the example with the modified versions
    # # item["input_text"] = modified_input
    # # item["text"] = modified_input + modified_reference
    # # item["reference"] = modified_reference
    # print(item["text"])
    # input()
    return item["text"]

def PSM(example):
    # Assuming your dataset contains a "text" field for the document
    document = example["response_j"]

    # Calculate the lengths of preﬁx, middle, and sufﬁx
    total_length = len(document)
    prefix_length = total_length // 3
    suffix_length = total_length // 3
    middle_length = total_length - prefix_length - suffix_length

    # Split the document into preﬁx, middle, and sufﬁx
    prefix = document[:prefix_length]
    suffix = document[prefix_length:prefix_length + suffix_length]
    middle = document[prefix_length + suffix_length:]
    # print("prefix", prefix)
    # Tokenize and add sentinel tokens
    prefix_tokens = tokenizer("<PRE> " + prefix[0], padding="max_length", max_length=prefix_length)
    suffix_tokens = tokenizer("<SUF> " + suffix[0], padding="max_length", max_length=suffix_length)
    middle_tokens = tokenizer("<MID> " + middle[0], padding="max_length", max_length=middle_length)

    prefix_tokens["input_ids"] = torch.tensor(prefix_tokens["input_ids"])
    suffix_tokens["input_ids"] = torch.tensor(suffix_tokens["input_ids"])
    middle_tokens["input_ids"] = torch.tensor(middle_tokens["input_ids"])

    print("sqws", prefix_tokens["input_ids"])
    # Concatenate the tokens in the desired order
    input_ids = torch.cat([prefix_tokens["input_ids"], suffix_tokens["input_ids"], middle_tokens["input_ids"]])

    # Prepare the input dictionary
    input_dict = {
        "input_ids": input_ids,
        "attention_mask": input_ids != tokenizer.pad_token_id,
        # Add any other relevant fields for your dataset
    }

    return input_dict




def query_fim(idx, argu_idx, response_tensors):

    response_tensors = response_tensors.cpu().detach().numpy().tolist()
    # prefix_token = 3866
    # middle_token = 13602
    # suffix_token = 37333
    # eot_token = 4720

    prefix_token = 31000
    suffix_token = 31001
    middle_token = 31002
    eot_token = 31003

    rest_att = 0.2
    # print("idx", idx)
    # print("argu_idx", argu_idx)


    # punctuation = [11, 13, 764, 837, 0, 30, 29847, 16317, 50256]

    if idx == 0:
        
        middle = response_tensors[argu_idx[idx] + 1:argu_idx[idx+1]+1]
        suffix = response_tensors[argu_idx[idx+1] + 1:]
        prefix = response_tensors[:argu_idx[idx]+1]
    else:
        middle = response_tensors[argu_idx[idx-1] + 1:argu_idx[idx]+1]
        suffix = response_tensors[argu_idx[idx] + 1:]
        prefix = response_tensors[:argu_idx[idx-1]+1]

    if idx == 0:
        # focus = prefix
        # attention_tensors = torch.tensor([1] * len(focus) + [rest_att] * len(middle) + [rest_att] * len(suffix))
        
        query_FIM = [middle_token] + middle + [suffix_token] + suffix + [prefix_token]
        original_FIM = [middle_token] + middle + [suffix_token] + suffix + [prefix_token] + prefix + [eot_token]
        attention_tensors = torch.tensor([rest_att] * len([middle_token] + middle + [suffix_token] + suffix + [prefix_token]) + [1] * (len(prefix)+1))

        # action_to_assign = middle + suffix
    elif idx == len(argu_idx) - 1: 
        # focus = suffix
        # attention_tensors = torch.tensor([rest_att] * len(prefix) + [rest_att] * len(middle) + [1] * len(focus))
        query_FIM = [prefix_token] + prefix + [middle_token] + middle + [suffix_token]
        original_FIM = [prefix_token] + prefix + [middle_token] + middle + [suffix_token] + suffix + [eot_token]
        attention_tensors = torch.tensor([rest_att] * len([prefix_token] + prefix + [middle_token] + middle + [suffix_token]) + [1] * (len(suffix)+1))
        # action_to_assign = 
    else:
        # focus = middle
        # attention_tensors = torch.tensor([rest_att] * len(prefix) + [1] * len(focus) + [rest_att] * len(suffix))
        query_FIM = [prefix_token] + prefix + [suffix_token] + suffix + [middle_token]
        original_FIM = [prefix_token] + prefix + [suffix_token] + suffix + [middle_token] + middle + [eot_token]
        attention_tensors = torch.tensor([rest_att] * len([prefix_token] + prefix + [suffix_token] + suffix + [middle_token]) + [1] * (len(middle)+1))
    # attention_tensors = attention_tensors.to("cuda:0")
    # # increase the dimension of response_tensors and attention_tensors
    # response_tensors = response_tensors.unsqueeze(dim=0)
    # # print("response_tensors", response_tensors)
    # # print("attention_tensors", attention_tensors)
    # attention_tensors = attention_tensors.unsqueeze(dim=0)
    # phrase_cost = cost_model(response_tensors, attention_tensors)[0]
    # # concatenate query_tensors and state_to_assign
    # # state_to_assign = torch.cat([query_tensors, state_to_assign])
    # # action_to_assign = 

    # # print("state_to_assign", state_to_assign)
    # # print("action_to_assign", action_to_assign)

    # return phrase_cost.cpu().detach().numpy()[0][0]
    query_FIM = torch.tensor(query_FIM)
    original_FIM = torch.tensor(original_FIM)
    
    return query_FIM, original_FIM, attention_tensors
    
    # print("phrase_cost", phrase_cost)
    # input("cost")
    
def query_inverse(query):
    
    prefix_token = 31000
    suffix_token = 31001
    middle_token = 31002
    eot_token = 31003
    
    # print("query", query)
    # input()
    # find the index of prefix_token in query
    prefix_idx = query.index(prefix_token)
    # find the index of suffix_token in query
    suffix_idx = query.index(suffix_token)
    # find the index of middle_token in query
    middle_idx = query.index(middle_token)
    
    if prefix_idx > middle_idx:
        prefix = query[prefix_idx:]
        middle = query[middle_idx:suffix_idx]
        suffix = query[suffix_idx:prefix_idx]
    elif suffix_idx > middle_idx:
        prefix = query[prefix_idx:middle_idx]
        middle = query[middle_idx:suffix_idx]
        suffix = query[suffix_idx:]
    else:
        prefix = query[prefix_idx:suffix_idx]
        suffix = query[suffix_idx:middle_idx]
        middle = query[middle_idx:]
        
 
    prefix = prefix[1:]
    middle = middle[1:]
    suffix = suffix[1:]
    
    inverse_res = prefix + middle + suffix
    return inverse_res
    
    # print("prefix", prefix)
    # print("middle", middle)
    # print("suffix", suffix)
    # input()
    



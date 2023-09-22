# import argparse
# import os

# from accelerate import Accelerator
# from datasets import load_dataset
# from peft import LoraConfig
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed

# from trl import SFTTrainer
# from trl.trainer import ConstantLengthDataset

# import random
# import warnings
# from collections import deque
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import IterableDataset
# """
# Fine-Tune Llama-7b on SE paired dataset
# """


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, default="")
#     # parser.add_argument("--dataset_name", type=str, default="lvwerra/stack-exchange-paired")
#     parser.add_argument("--dataset_name", type=str, default="wics/strategy-qa")
#     parser.add_argument("--subset", type=str, default="data/finetune")
#     parser.add_argument("--split", type=str, default="test")
#     parser.add_argument("--size_valid_set", type=int, default=4000)
#     parser.add_argument("--streaming", action="store_true")
#     parser.add_argument("--shuffle_buffer", type=int, default=5000)

#     parser.add_argument("--seq_length", type=int, default=1024)
#     parser.add_argument("--max_steps", type=int, default=10000)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
#     parser.add_argument("--eos_token_id", type=int, default=49152)

#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
#     parser.add_argument("--num_warmup_steps", type=int, default=100)
#     parser.add_argument("--weight_decay", type=float, default=0.05)

#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument("--no_fp16", action="store_false")
#     parser.add_argument("--bf16", action="store_true", default=False)
#     parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--num_workers", type=int, default=None)
#     parser.add_argument("--output_dir", type=str, default="./checkpoints")
#     parser.add_argument("--log_freq", default=1, type=int)
#     parser.add_argument("--eval_freq", default=1000, type=int)
#     parser.add_argument("--save_freq", default=1000, type=int)

#     return parser.parse_args()


# def chars_token_ratio(dataset, tokenizer, nb_examples=400):
#     """
#     Estimate the average number of characters per token in the dataset.
#     """
#     total_characters, total_tokens = 0, 0
#     for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
#         text = prepare_sample_text(example)
#         total_characters += len(text)
#         if tokenizer.is_fast:
#             total_tokens += len(tokenizer(text).tokens())
#         else:
#             total_tokens += len(tokenizer.tokenize(text))

#     return total_characters / total_tokens


# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )


# def prepare_sample_text(item):
#     """Prepare the text from a sample of the dataset."""
#     # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
#     # return text
#     if item["answer"] == True:
#         judge = "Yes."
#     else:
#         judge = "No."

#     reference = judge + ' '
    
#     for idx, i in enumerate(item["facts"]):
#         if i[-1] != '.':
#             i = i + '.'
#         order = ""
#         if idx == 0:
#             order = "Firstly, "
#         if idx == 1:
#             order = "Secondly, "
#         if idx == 2:
#             order = "Thirdly, "
#         if idx == 3:
#             order = "Forthly, "

#         # if idx == len(item["facts"]) - 1:
#         #     order = "Finally, "
#         reference = reference + order + i + ' '
    
#     reference = reference + "Thus the answer is " + judge
            
#     # Modify the input text and its corresponding reference
#     modified_input = item["question"]
#     text = modified_input + ' ' + reference
    
#     # modified_input = "The answer to the question: " + modified_input + "is " + judge + " Because"
#     # item["question"] = modified_input
#     # Update the example with the modified versions
#     # item["input_text"] = modified_input
#     # item["text"] = modified_input + modified_reference
#     # item["reference"] = modified_reference

#     return text

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


# if getattr(tokenizer, "pad_token", None) is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # special_tokens = ["<PRE>", "<SUF>", "<MID>", "<EOT>"]
# # tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
# # tokenizer.add_tokens(special_tokens)

# # text = "This is an example <PRE> text <MID> with special <SUF> tokens. <EOT>"
# # tokens = tokenizer.encode(text, add_special_tokens=True)


# # print("tokens", tokens)
# # decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
# # print("decoded text", decoded_text)
# # input()

# def PSM(example):
#     # Assuming your dataset contains a "text" field for the document
#     document = example["response_j"]

#     # Calculate the lengths of preﬁx, middle, and sufﬁx
#     total_length = len(document)
#     prefix_length = total_length // 3
#     suffix_length = total_length // 3
#     middle_length = total_length - prefix_length - suffix_length

#     # Split the document into preﬁx, middle, and sufﬁx
#     prefix = document[:prefix_length]
#     suffix = document[prefix_length:prefix_length + suffix_length]
#     middle = document[prefix_length + suffix_length:]
#     # print("prefix", prefix)
#     # Tokenize and add sentinel tokens
#     prefix_tokens = tokenizer("<PRE> " + prefix[0], padding="max_length", max_length=prefix_length)
#     suffix_tokens = tokenizer("<SUF> " + suffix[0], padding="max_length", max_length=suffix_length)
#     middle_tokens = tokenizer("<MID> " + middle[0], padding="max_length", max_length=middle_length)

#     prefix_tokens["input_ids"] = torch.tensor(prefix_tokens["input_ids"])
#     suffix_tokens["input_ids"] = torch.tensor(suffix_tokens["input_ids"])
#     middle_tokens["input_ids"] = torch.tensor(middle_tokens["input_ids"])

#     print("sqws", prefix_tokens["input_ids"])
#     # Concatenate the tokens in the desired order
#     input_ids = torch.cat([prefix_tokens["input_ids"], suffix_tokens["input_ids"], middle_tokens["input_ids"]])

#     # Prepare the input dictionary
#     input_dict = {
#         "input_ids": input_ids,
#         "attention_mask": input_ids != tokenizer.pad_token_id,
#         # Add any other relevant fields for your dataset
#     }

#     return input_dict


# class ConstantLengthDataset(IterableDataset):
#     """
#     Iterable dataset that returns constant length chunks of tokens from stream of text files.
#     The dataset also formats the text before tokenization with a specific format that is provided
#     by the user.

#         Args:
#             tokenizer (`transformers.PreTrainedTokenizer`):
#                 The processor used for processing the data.
#             dataset (`dataset.Dataset`):
#                 Dataset with text files.
#             dataset_text_field (`str`, **optional**):
#                 Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
#             formatting_func (`Callable`, **optional**):
#                 Function that formats the text before tokenization. Usually it is recommended to have follows a certain
#                 pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
#             infinite (`bool`, *optional*, defaults to `False`):
#                 If True the iterator is reset after dataset reaches end else stops.
#             seq_length (`int`, *optional*, defaults to `1024`):
#                 Length of token sequences to return.
#             num_of_sequences (`int`, *optional*, defaults to `1024`):
#                 Number of token sequences to keep in buffer.
#             chars_per_token (`int`, *optional*, defaults to `3.6`):
#                 Number of characters per token used to estimate number of tokens in text buffer.
#             eos_token_id (`int`, *optional*, defaults to `0`):
#                 Id of the end of sequence token if the passed tokenizer does not have an EOS token.
#             shuffle ('bool', *optional*, defaults to True)
#                 Shuffle the examples before they are returned
#     """

#     def __init__(
#         self,
#         tokenizer,
#         dataset,
#         dataset_text_field=None,
#         formatting_func=None,
#         infinite=False,
#         seq_length=1024,
#         num_of_sequences=1024,
#         chars_per_token=3.6,
#         eos_token_id=0,
#         shuffle=True,
#     ):
#         self.tokenizer = tokenizer

#         if tokenizer.eos_token_id is None:
#             warnings.warn(
#                 "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
#                 f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
#             )

#         self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
#         self.dataset = dataset
#         self.seq_length = seq_length
#         self.infinite = infinite
#         self.current_size = 0
#         self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
#         self.shuffle = shuffle
#         if formatting_func is None:
#             self.formatting_func = lambda x: x[dataset_text_field]
#         else:
#             self.formatting_func = formatting_func

#         if formatting_func is not None:
#             formatting_func_signature = formatting_func.__code__.co_varnames
#             if len(formatting_func_signature) > 1:
#                 warnings.warn(
#                     "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
#                     " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
#                 )

#     def __len__(self):
#         return len(self.dataset)

#     def __iter__(self):
#         iterator = iter(self.dataset)
#         more_examples = True
#         while more_examples:
#             buffer, buffer_len = [], 0
#             while True:
#                 if buffer_len >= self.max_buffer_size:
#                     break
#                 try:
#                     buffer.append(self.formatting_func(next(iterator)))
#                     # print("buffer", buffer)
#                     # input()
#                     buffer_len += len(buffer[-1])
#                 except StopIteration:
#                     if self.infinite:
#                         iterator = iter(self.dataset)
#                         warnings.warn("The dataset reached end and the iterator is reset to the start.")
#                     else:
#                         more_examples = False
#                         break
#             tokenized_inputs = self.tokenizer(buffer, truncation=False, add_special_tokens=True)["input_ids"]
#             #  find the index of element in tokenized_inputs that equals 29889
#             # special sentinel tokens
#             # ["<PRE>", "<SUF>", "<MID>"]
#             # <PRE>: 32000
#             # <MID>: 32002
#             # <SUF>: 32001
#             prefix_token = 31000
#             suffix_token = 31001
#             middle_token = 31002
#             eot_token = 31003
#             for tkidx, token_input in enumerate(tokenized_inputs):
#                 argu_idx = []
#                 for idx, token in enumerate(token_input):
#                     if token == 29889:
#                         argu_idx.append(idx)
#                 # print("token_input", token_input)
#                 # print("argu_idx", argu_idx)
#                 # randomly choose 3 elements from argu_idx in assending order
#                 if len(argu_idx) > 3:
#                     argu_idx = random.sample(argu_idx[1:], 3)

#                 argu_idx.sort()
#                 # print("argu_idx", argu_idx)
#                 prefix = token_input[:argu_idx[0]+1]
#                 suffix = token_input[argu_idx[1] + 1:]
#                 middle = token_input[argu_idx[0] + 1:argu_idx[1]+1]
#                 input_psm_transform = [prefix[0]] + [prefix_token] + prefix[1:] + [suffix_token] + suffix + [middle_token] + middle + [eot_token]
#                 tokenized_inputs[tkidx] = input_psm_transform
#                 # print("prefix", prefix)
#                 # print("suffix", suffix)
#                 # print("middle", middle)
#                 # print("input_psm_transform", input_psm_transform)
#                 # decoded_text = tokenizer.decode(input_psm_transform)#, skip_special_tokens=True)
#                 # print("decoded text", decoded_text)
                
#                 # input()

#             all_token_ids = []
#             for tokenized_input in tokenized_inputs:
#                 all_token_ids.extend(tokenized_input + [self.concat_token_id])
#                 # print("all_token_ids", all_token_ids)
#                 # input()
#             examples = []
#             for i in range(0, len(all_token_ids), self.seq_length):
#                 input_ids = all_token_ids[i : i + self.seq_length]
#                 if len(input_ids) == self.seq_length:
#                     examples.append(input_ids)
#             if self.shuffle:
#                 random.shuffle(examples)
#             for example in examples:
#                 self.current_size += 1
#                 yield {
#                     "input_ids": torch.LongTensor(example),
#                     "labels": torch.LongTensor(example),
#                 }
#                 # print("example", example)


# def create_datasets(tokenizer, args):
#     dataset = load_dataset(
#         args.dataset_name,
#         data_dir=args.subset,
#         split=args.split,
#         use_auth_token=True,
#         num_proc=args.num_workers if not args.streaming else None,
#         streaming=args.streaming,
#     )

#     print("dataset", dataset)
#     # Assuming you have a Hugging Face dataset called "my_dataset"
#     # dataset = dataset.map(PSM, batched=True)
#     # print("new dataset", dataset)
#     # input()

#     if args.streaming:
#         print("Loading the dataset in streaming mode")
#         valid_data = dataset.take(args.size_valid_set)
#         train_data = dataset.skip(args.size_valid_set)
#         train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
#     else:
#         dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
#         train_data = dataset["train"]
#         valid_data = dataset["test"]
#         print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

#     chars_per_token = chars_token_ratio(train_data, tokenizer)
#     print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

#     train_dataset = ConstantLengthDataset(
#         tokenizer,
#         train_data,
#         formatting_func=prepare_sample_text,
#         infinite=True,
#         seq_length=args.seq_length,
#         chars_per_token=chars_per_token,
#     )
#     valid_dataset = ConstantLengthDataset(
#         tokenizer,
#         valid_data,
#         formatting_func=prepare_sample_text,
#         infinite=False,
#         seq_length=args.seq_length,
#         chars_per_token=chars_per_token,
#     )
#     return train_dataset, valid_dataset


# def run_training(args, train_data, val_data):
#     print("Loading the model")

#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

#     train_data.start_iteration = 0

#     print("Starting main loop")

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         dataloader_drop_last=True,
#         evaluation_strategy="steps",
#         max_steps=args.max_steps,
#         eval_steps=args.eval_freq,
#         save_steps=args.save_freq,
#         logging_steps=args.log_freq,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         learning_rate=args.learning_rate,
#         lr_scheduler_type=args.lr_scheduler_type,
#         warmup_steps=args.num_warmup_steps,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         gradient_checkpointing=not args.no_gradient_checkpointing,
#         fp16=not args.no_fp16,
#         bf16=args.bf16,
#         weight_decay=args.weight_decay,
#         run_name="llama-7b-finetuned",
#         report_to="tensorboard",
#         ddp_find_unused_parameters=False,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
#     )

#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=val_data,
#         peft_config=lora_config,
#         packing=True,
#     )

#     print_trainable_parameters(trainer.model)

#     print("Training...")
#     trainer.train()

#     print("Saving last checkpoint of the model")
#     trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


# def main(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     train_dataset, eval_dataset = create_datasets(tokenizer, args)
#     run_training(args, train_dataset, eval_dataset)


# if __name__ == "__main__":
#     args = get_args()
#     assert args.model_path != "", "Please provide the llama model path"

#     set_seed(args.seed)
#     os.makedirs(args.output_dir, exist_ok=True)

#     logging.set_verbosity_error()

#     main(args)

import argparse
import os, sys
sys.path.append("/home/gong112/service_backup/work/zhaorun/SS-RLHF")

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
import sys
sys.path.append("/home/xyq/.conda/trl/")
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
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
"""
Fine-Tune Llama-7b on SE paired dataset
"""

os.environ['CURL_CA_BUNDLE'] = ''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/xyq/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/")
    # parser.add_argument("--dataset_name", type=str, default="lvwerra/stack-exchange-paired")
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--subset", type=str, default="data/finetune")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()


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

tokenizer = AutoTokenizer.from_pretrained("/home/xyq/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/")

# tokenizer = AutoTokenizer.from_pretrained("/home/xyq/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/")
# tokenizer = AutoTokenizer.from_pretrained("/home/xyq/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9")  #  ("lvwerra/gpt2-imdb")

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# special_tokens = ["<PRE>", "<SUF>", "<MID>", "<EOT>"]
# tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
# tokenizer.add_tokens(special_tokens)

# text = "This is an example <PRE> text <MID> with special <SUF> tokens. <EOT>"
# tokens = tokenizer.encode(text, add_special_tokens=True)


# print("tokens", tokens)
# decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
# print("decoded text", decoded_text)
# input()


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )

    # print("dataset", dataset)
    # Assuming you have a Hugging Face dataset called "my_dataset"
    # dataset = dataset.map(PSM, batched=True)
    # print("new dataset", dataset)
    # input()

    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
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
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="gpt2-imdb-finetuned",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    # model = AutoModelForCausalLM.from_pretrained(
    # args.model_path
    # )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)

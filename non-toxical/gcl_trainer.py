import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, pipeline
from transformers import HfArgumentParser, PreTrainedTokenizerBase, Trainer, AutoModelForSequenceClassification
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
from transformers.utils import PaddingStrategy
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

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2-imdb",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=5,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=2,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# decive = torch.device("cpu")
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

dataset = load_dataset(
    dataset_name,
    data_dir="data/finetune",
    split="train",
    use_auth_token=True,
    num_proc=1
    # streaming=args.streaming,
)

print("dataset", dataset)

dataset = dataset.train_test_split(test_size=0.1, seed=0)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

train_dataset = train_dataset.select(range(300))
eval_dataset = eval_dataset.select(range(5))

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 16,
}

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

original_columns = train_dataset.column_names
tokenized_max_length = 128

def preprocess_function(examples):
    # print("handling one example")
    # split examples (shape 1000) into ten pieces of each one (shape 100)
    # examples is a string
    # print(examples['text'])
    # shaped_examples = []
    # for i in range(10):
    #     shaped_examples.append(examples['text'][i*100:(i+1)*100])
    
    # print("length", len(examples['text']))
    # input()
    # print("shaped_examples", len(shaped_examples))
    new_examples = {
        "samp_input_ids": [],
        "samp_attention_mask": [],
        "demo_input_ids": [],
        "demo_attention_mask": [],
    }

    # for text in shaped_examples:
    # print("text", text)
    # input()
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=tokenized_max_length)["input_ids"]

    # print("tokenized_inputs", tokenized_inputs)
    # print("tokenized_inputs", tokenized_inputs)
    # input()

    query_tensors = torch.LongTensor(tokenized_inputs)

    # print("query_tensors", query_tensors)

    
    argu_idx = []
    # print("query_tensors", query_tensors)
    # if it is a list
    # if type(query_tensors) == list:
    #     print("query_tensors", query_tensors)

    query_text = tokenizer.decode(query_tensors)
    # print("decoded_text", query_text)

    pipe_outputs = sentiment_pipe(query_text, **sent_kwargs)

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

    for idx in tqdm(range(len(argu_idx)-1), total=len(argu_idx)):


        phrase = argu_idx[idx]

        middle = query_tensors[argu_idx[idx] + 1:argu_idx[idx+1]+1]
        suffix = query_tensors[argu_idx[idx+1] + 1:]
        prefix = query_tensors[:argu_idx[idx]+1]

        start_special = [prefix[0]]
        prefix = prefix[1:]

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
        # print("D_samp", len(D_samp))
        # print("D_demo", len(D_demo))
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
        new_examples["samp_input_ids"].append(D_samp[idx].numpy())
        new_examples["samp_attention_mask"].append(D_samp_attention[idx].numpy())

    demo_features = []
    for idx, d_samp in enumerate(D_demo):
        new_examples["demo_input_ids"].append(D_demo[idx].numpy())
        new_examples["demo_attention_mask"].append(D_demo_attention[idx].numpy())

        # print("new_examples", new_examples)
    
    return new_examples

num_proc = 1
input_max_length = 256

# # print("train_dataset", train_dataset)
# # print("eval_dataset", eval_dataset)
train_dataset = train_dataset.map(
    preprocess_function,
    batched=False,
    num_proc=num_proc,
    remove_columns=original_columns,
)
train_dataset.save_to_disk("gcl/train_dataset_bigger")

# input("save to disk")
# train_dataset = load_from_disk("gcl/train_dataset")

train_dataset = train_dataset.filter(
    lambda x: len(x["samp_input_ids"]) <= input_max_length and len(x["demo_input_ids"]) <= input_max_length
)

# eval_dataset = eval_dataset.map(
#     preprocess_function,
#     batched=False,
#     num_proc=num_proc,
#     remove_columns=original_columns,
# )
# eval_dataset = eval_dataset.filter(
#     lambda x: len(x["samp_input_ids"]) <= input_max_length and len(x["demo_input_ids"]) <= input_max_length
# )



# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        samp_features = []
        demo_features = []
        for feature in features:
            # print("feature", np.shape(feature["samp_input_ids"][0]))
            # print("feature", np.shape(feature["samp_input_ids"][2]))
            # print("feature", np.shape(feature["samp_input_ids"][-1]))
            # input()
            for i in range(len(feature["samp_input_ids"])):
                samp_features.append(
                    {
                        "input_ids": feature["samp_input_ids"][i],
                        "attention_mask": feature["samp_attention_mask"][i],
                    })

            for j in range(len(feature["demo_input_ids"])):
                demo_features.append(
                    {
                        "input_ids": feature["demo_input_ids"][j],
                        "attention_mask": feature["demo_attention_mask"][j],
                    })

        
        samp_batch = tokenizer.pad(
            samp_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
       
        
        demo_batch = tokenizer.pad(
            demo_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "samp_input_ids": samp_batch["input_ids"],
            "samp_attention_mask": samp_batch["attention_mask"],
            "demp_input_ids": demo_batch["input_ids"],
            "demo_attention_mask": demo_batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# # Define the metric that we'll use for validation.
# accuracy = evaluate.load("accuracy")


# def compute_metrics(eval_pred):
#     predictions, _ = eval_pred
#     # Here, predictions is rewards_j and rewards_k.
#     # We want to see how much of the time rewards_j > rewards_k.
#     predictions = np.argmax(predictions, axis=0)
#     labels = np.zeros(predictions.shape)
#     return accuracy.compute(predictions=predictions, references=labels)

# cost_f = cost_model
# cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)

class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        costs_samp = model(input_ids=inputs['samp_input_ids'], attention_mask=inputs['samp_attention_mask'])[0]
        costs_demo = model(input_ids=inputs['demo_input_ids'], attention_mask=inputs['demo_attention_mask'])[0]

        # print("costs_demo", costs_demo)
        probs = torch.ones(len(costs_samp))
        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
            
        # UPDATING THE COST FUNCTION
        # cost_optimizer.zero_grad()
        # loss_IOC.backward()  
        # cost_optimizer.step()
        # # print("loss_IOC", loss_IOC)
        return loss_IOC

output_name = "gpt2-imdb-cost-assignment"

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    # eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)


# Train the model, woohoo.
trainer = RewardTrainer(
    model=cost_model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=128),
)


# if script_args.eval_first_step:

#     class EvaluateFirstStepCallback(TrainerCallback):
#         def on_step_end(self, args, state, control, **kwargs):
#             if state.global_step == 1:
#                 control.should_evaluate = True

#     trainer.add_callback(EvaluateFirstStepCallback())

trainer.train(script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
model.save_pretrained(output_name + "_peft_last_checkpoint")











# def create_datasets(tokenizer):
#     dataset = load_dataset(
#         dataset_name,
#         data_dir="data/finetune",
#         split="test",
#         use_auth_token=True,
#         num_proc=1
#         # streaming=args.streaming,
#     )

#     print("dataset", dataset)

#     dataset = dataset.train_test_split(test_size=0.1, seed=0)
#     train_data = dataset["train"]
#     valid_data = dataset["test"]
#     print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

#     chars_per_token = chars_token_ratio(train_data, tokenizer)
#     print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

#     train_dataset = ConstantLengthDataset(
#         tokenizer,
#         train_data,
#         formatting_func=prepare_sample_text,
#         infinite=True,
#         seq_length=128,
#         chars_per_token=chars_per_token,
#     )
#     valid_dataset = ConstantLengthDataset(
#         tokenizer,
#         valid_data,
#         formatting_func=prepare_sample_text,
#         infinite=False,
#         seq_length=128,
#         chars_per_token=chars_per_token,
#     )
#     return train_dataset, valid_dataset



# # Replace the last layer of the GPT-2 model with your custom cost function
# # model.transformer.h[-1] = RewardFunction(768, 1)
# train_dataset, valid_dataset = create_datasets(tokenizer)


# for epoch, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):
#     query_tensors = batch["input_ids"]
#     argu_idx = []
#     # print("query_tensors", query_tensors)
#     query_text = tokenizer.decode(query_tensors)
#     print("decoded_text", query_text)

#     pipe_outputs = sentiment_pipe(query_text, **sent_kwargs)

#     print("label reward: ", pipe_outputs)

#     # query_tensors = query_tensors.tonumpy()
#     # convert to numpy
#     query_tensors = query_tensors.numpy().tolist()


#     for idx, token_input in enumerate(query_tensors):
#         #if token == 29889:
        
#         prefix_token = 3866
#         middle_token = 13602
#         suffix_token = 37333
#         eot_token = 4720
#         punctuation = [11, 13, 764, 837, 0, 30, 29847, 16317, 50256]
#         # print("token_input", token_input)

#         if token_input in punctuation:
#             argu_idx.append(idx)

#         # if len(argu_idx) >= 2:
#         #     argu_idx = random.sample(argu_idx[1:], 2)

#     argu_idx.sort()

#     samp_num = 50
#     rest_att = 0.2
#     best_of_rwd = 5
#     for idx in range(len(argu_idx)-1):
#         phrase = argu_idx[idx]

#         D_samp = []
#         D_samp_texts = []
#         D_samp_attention = []
#         D_samp_index = []

    
#         middle = query_tensors[argu_idx[idx] + 1:argu_idx[idx+1]+1]
#         suffix = query_tensors[argu_idx[idx+1] + 1:]
#         prefix = query_tensors[:argu_idx[idx]+1]

#         D_samp.append(torch.tensor(query_tensors))
#         D_samp_texts.append(query_text)

#         if idx == 0: # first phrase is doing gcl
#             D_samp_attention.append(torch.tensor([1] * len(prefix) + [rest_att] * len(middle) + [rest_att] * len(suffix)))

#             D_samp_index.append([0, len(prefix)-1])
#             fim_transform_query = [middle_token] + middle + [suffix_token] + suffix + [prefix_token]
#             expert_response = prefix
#             fim_transform_query = torch.tensor(fim_transform_query)

#             for s in range(samp_num):
#                 response = model.generate(input_ids=fim_transform_query.unsqueeze(dim=0), **generation_kwargs)
#                 response_tensors = response[:, fim_transform_query.shape[0] :]

#                 sample_tensors = response_tensors[0].tolist() + middle + suffix
#                 # attention_tensors should be all 1 where response_tensors[0].tolist() is, and 0.5 at other elements

#                 attention_tensors = torch.tensor([1] * len(response_tensors[0].tolist()) + [rest_att] * len(middle) + [rest_att] * len(suffix))
#                 sample_tensors = torch.tensor(sample_tensors)
#                 sample_texts = tokenizer.decode(sample_tensors)
#                 # print("sample_texts", len(response_tensors[0].tolist()))

#                 D_samp.append(sample_tensors)
#                 D_samp_texts.append(sample_texts)
#                 D_samp_attention.append(attention_tensors)

#                 fill_index = [0, len(response_tensors[0].tolist())-1]

#                 D_samp_index.append(fill_index)

#                 # print("sample_texts", sample_texts)
#                 # input()
#                 # pipe_outputs = sentiment_pipe(sample_texts, **sent_kwargs)

#         elif idx == len(argu_idx) - 1: # last phrase is doing gcl
#             D_samp_attention.append(torch.tensor([rest_att] * len(prefix) + [rest_att] * len(middle) + [1] * len(suffix)))
#             D_samp_index.append([len(prefix + middle), len(query_tensors)-1])
#             fim_transform_query = [prefix_token] + prefix + [middle_token] + middle + [suffix_token]
#             expert_response = suffix
#             fim_transform_query = torch.tensor(fim_transform_query)

#             for s in range(samp_num):
#                 response = model.generate(input_ids=fim_transform_query.unsqueeze(dim=0), **generation_kwargs)
#                 response_tensors = response[:, fim_transform_query.shape[0] :]

#                 sample_tensors = prefix + middle + response_tensors[0].tolist()
#                 sample_texts = tokenizer.decode(sample_tensors)
#                 sample_tensors = torch.tensor(sample_tensors)
#                 D_samp.append(sample_tensors)
#                 D_samp_texts.append(sample_texts)
#                 D_samp_attention.append(torch.tensor([rest_att] * len(prefix) + [rest_att] * len(middle) + [1] * len(response_tensors[0].tolist())))
#                 fill_index = [len(prefix + middle), len(sample_tensors)-1]

#                 D_samp_index.append(fill_index)

#         else:
#             D_samp_attention.append(torch.tensor([rest_att] * len(prefix) + [1] * len(middle) + [rest_att] * len(suffix)))
#             D_samp_index.append([len(prefix), len(prefix + middle)-1])
#             fim_transform_query = [prefix_token] + prefix + [suffix_token] + suffix + [middle_token]
#             expert_response = middle
#             fim_transform_query = torch.tensor(fim_transform_query)

#             for s in range(samp_num):
#                 response = model.generate(input_ids=fim_transform_query.unsqueeze(dim=0), **generation_kwargs)
#                 response_tensors = response[:, fim_transform_query.shape[0] :]

#                 sample_tensors = prefix + response_tensors[0].tolist() + suffix

#                 sample_texts = tokenizer.decode(sample_tensors)
#                 sample_tensors = torch.tensor(sample_tensors)
#                 D_samp.append(sample_tensors)
#                 D_samp_texts.append(sample_texts)
#                 D_samp_attention.append(torch.tensor([rest_att] * len(prefix) + [1] * len(response_tensors[0].tolist()) + [rest_att] * len(suffix)))

#                 fill_index = [len(prefix), len(prefix + response_tensors[0].tolist())-1]

#                 D_samp_index.append(fill_index)


#         ### guided cost learning ###

#         # print(D_samp)
#         # print(len(D_samp))
#         reward_list = sentiment_pipe(D_samp_texts, **sent_kwargs)
#         # find the index of 3 largest reward in reward_list according to reward_list[1]['score]
#         max_rew_idx = sorted(range(len(reward_list)), key=lambda i: reward_list[i][1]['score'])[-best_of_rwd:]
#         # print("max_rew_idx", max_rew_idx)
#         D_demo = []
#         D_demo_attention = []
#         for bd in max_rew_idx:
#             D_demo.append(D_samp[bd])
#             D_demo_attention.append(D_samp_attention[bd])

#         samp_features = []
#         for idx, d_samp in enumerate(D_samp):
#             # input_ids = torch.tensor(D_samp[idx])
#             # attention_mask = torch.tensor(D_samp_attention[idx])
#             # print(D_samp[idx])
#             samp_features.append(
#                 {
#                     "input_ids": D_samp[idx].numpy(),
#                     "attention_mask": D_samp_attention[idx].numpy(),
#                 })


#         samp_batch = tokenizer.pad(
#             samp_features,
#             padding=True,
#             max_length=128,
#             pad_to_multiple_of=None,
#             return_tensors="pt",
#         )

#         # print("batch", batch)

        
#         # print("costs_samp", costs_samp)
#         # input("cost generated for a batch")
#         demo_features = []
#         for idx, d_samp in enumerate(D_demo):
#             # input_ids = torch.tensor(D_samp[idx])
#             # attention_mask = torch.tensor(D_samp_attention[idx])
#             # print(D_samp[idx])
#             demo_features.append(
#                 {
#                     "input_ids": D_demo[idx].numpy(),
#                     "attention_mask": D_demo_attention[idx].numpy(),
#                 })


#         demo_batch = tokenizer.pad(
#             demo_features,
#             padding=True,
#             max_length=128,
#             pad_to_multiple_of=None,
#             return_tensors="pt",
#         )

#         for i in range(10):
#             costs_samp = cost_f(samp_batch['input_ids'], samp_batch['attention_mask'])[0]
#             costs_demo = cost_f(demo_batch['input_ids'], demo_batch['attention_mask'])[0]

#             # print("costs_demo", costs_demo)
#             probs = torch.ones(len(costs_samp))
#             # LOSS CALCULATION FOR IOC (COST FUNCTION)
#             loss_IOC = torch.mean(costs_demo) + \
#                     torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
                
#             # UPDATING THE COST FUNCTION
#             cost_optimizer.zero_grad()
#             loss_IOC.backward()  
#             cost_optimizer.step()
#             print("loss_IOC", loss_IOC)
            
#         print("next fim")
#         # print(reward_list)



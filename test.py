# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2Model


# model_name = '/home/xyq/.conda/SS-RLHF/model/gpt2-finetuned'
model_name = '/home/xyq/.conda/trl/model/llama-finetuned'
tokenizer = AutoTokenizer.from_pretrained(model_name)#.to("cuda:1")
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:1")
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2Model.from_pretrained(model_name)

# prefix: ĥ 225
# suffix: Ħ 226
# middle: ħ 227
# eot: Ĩ 228


query_text = "<PRE> Can a man give birth to a child? <SUF> Thus the answer is no. <MID>"

query_tensors = tokenizer(query_text, return_tensors="pt").to("cuda:1")

# output = model(**query_tensors)
for i in range(10):
    response_tensors = model.generate(**query_tensors, max_new_tokens=128)



# print("output", output)
# print("response_tensors", response_tensors)

    decoded_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
    print("decoded text", decoded_text)
import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
paddle.set_device('gpu:3')
# input_s = "The meaning of life is" # The meaning of life is China is a great country.

input_s = f"<start_of_turn>user\nwrite a quick sort in python<end_of_turn>\n<start_of_turn>model\n"

model_name = "google/gemma-2b"
# model_name = "facebook/llama-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=paddle.float16, use_cache=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model.eval()
breakpoint()
generate_ids = model.generate(**tokenizer(input_s, return_tensors="pd"), decode_strategy="sampling", max_length=200)
print(tokenizer.decode(generate_ids[0][0]))
# model.sample(**tokenizer(input_s, return_tensors="pd"), max_length=20, pad_token_id=0, eos_token_id=1)
breakpoint()
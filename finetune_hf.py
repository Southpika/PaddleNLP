import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
deepspeed.init_distributed()

import os
import torch
import transformers
import time
# from accelerate import Accelerator, PartialState
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_logger(path):
    log_file = os.path.join(path, 'logfile.log')
    logger = logging.getLogger('my_logger')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# device_string = PartialState().process_index 
# accelerator = Accelerator(
#     cpu=None, mixed_precision=None, log_with="all", project_dir="./output/log"
# )

model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
print(model.device)
# Load the dataset and format it for training. , device_map={'':device_string}
data = load_dataset("/root/tzh_workspace/PaddleNLP/llm/data/")
def tokenize_function(example):
    res = [example["src"][i]+example["tgt"][i] for i in range(len(example["src"]))]
    return tokenizer(res, padding="max_length", truncation=True, max_length=1024)

train_dataset = data["train"].select(range(1000)).map(tokenize_function, batched=True)
eval_dataset = data["validation"].map(tokenize_function, batched=True)

print(train_dataset[0]["input_ids"].__len__())

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     dataset_text_field="src",
#     packing=False,
#     max_seq_length = 256,
#     args=TrainingArguments(
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=8,
#         warmup_steps=10,
#         num_train_epochs=1,
#         learning_rate=3e-6,
#         fp16=True,
#         logging_steps=1,
#         logging_dir='./output',
#         optim="adamw_torch",
#         evaluation_strategy="epoch",
#         save_total_limit=3,
#         output_dir='./output',
#         gradient_checkpointing=True,
#         lr_scheduler_type="linear", 
#         warmup_ratio=0.1, 
#     )
# )
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="adamw_torch", 
        logging_steps=1, 
        learning_rate=3e-5, 
        fp16=True,
        lr_scheduler_type="linear", 
        max_steps=50,
        save_strategy="steps", 
        output_dir='./output',
        gradient_checkpointing=False,
        fp16_opt_level="O2",
        deepspeed="/root/tzh_workspace/PaddleNLP/ds_config_stage3.json"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
logger = get_logger("/root/tzh_workspace/use_time")
cur_time = time.time()
logger.info(f"[Start of Training if HF]{cur_time}, use data size {len(train_dataset)}")
trainer.train()
end_time = time.time()
logger.info(f"[End of Training if HF]{cur_time}, use {end_time-cur_time} s")

import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTTrainer

from accelerate import PartialState
device_string = PartialState().process_index 
accelerator = Accelerator(
    cpu=None, mixed_precision=None, log_with="all", project_dir="./output/log"
)

model_id = "google/gemma-2b"


# Load the pretrained model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={'':device_string})
# model.to(accelerator.device)
# model.is_parallelizable = True
# model.model_parallel = True
print(model.device)
# breakpoint()
# Load the dataset and format it for training.
data = load_dataset("/root/tzh_workspace/PaddleNLP/llm/data/")
def tokenize_function(example):
    return tokenizer(example["src"], padding="max_length", truncation=True)
train_dataset = data["train"].map(tokenize_function, batched=True)
eval_dataset = data["validation"].map(tokenize_function, batched=True)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="src",
    packing=True,
    args=TrainingArguments(
        # fsdp_config=fsdp_config,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=30,
        num_train_epochs=1,
        learning_rate=3e-6,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=3,
        output_dir='./output',
        # fsdp="full_shard",
    )
)
# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=train_dataset,
#     args=transformers.TrainingArguments(
#         full_shard=True,
#         shard_grad_op=3,
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=16,
#         warmup_steps=30,
#         num_train_epochs=1,
#         learning_rate=3e-6,
#         # fp16=True,
#         logging_steps=1,
#         optim="adamw_torch",
#         evaluation_strategy="steps",
#         eval_steps=10,
#         save_total_limit=3,
#         output_dir='./output'
#     ),
#     data_collator=transformers.DataCollatorForSeq2Seq(
#             tokenizer,return_tensors="pt", padding=True
#         ),
# )
trainer.train()

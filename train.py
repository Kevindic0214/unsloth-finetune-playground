# train.py
from huggingface_hub import login
import wandb
from dotenv import load_dotenv
import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# === 環境變數與 wandb 登入 ===
load_dotenv()
wb_token = os.getenv("WANDB_TOKEN")
if wb_token:
    wandb.login(key=wb_token)
else:
    raise ValueError("WANDB_TOKEN not found in .env file.")

wandb.init(
    project="Fine-tuning-DeepSeek-R1-Distill-Qwen-7B on medical COT dataset",
    job_type="finetuning",
    anonymous="allow"
)

# === 模型與 tokenizer 載入 ===
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# === Prompt 格式 ===
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = [train_prompt_style.format(i, c, o) + EOS_TOKEN for i, c, o in zip(inputs, cots, outputs)]
    return {"text": texts}

# === 資料載入與格式化 ===
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'en', split="train[0:500]", trust_remote_code=True)
dataset = dataset.map(formatting_prompts_func, batched=True)

# === LoRA 微調準備 ===
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# === 訓練設定與啟動 ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
    ),
)

trainer.train()

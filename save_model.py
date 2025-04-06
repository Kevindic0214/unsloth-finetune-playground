# save_model.py
import os
from unsloth import FastLanguageModel

# === 設定 LoRA 微調過的模型輸出目錄 ===
output_dir = "outputs"

# === 載入微調後的模型（與 train.py 一致） ===
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# === 套用 LoRA 訓練結果 ===
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# === 儲存微調後的模型 ===
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"✅ LoRA fine-tuned model and tokenizer saved to `{output_dir}`.")

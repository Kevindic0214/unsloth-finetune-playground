# 🧪 unsloth-finetune-playground

This repo is my personal playground for experimenting with [Unsloth](https://github.com/unslothai/unsloth), a lightweight and blazing-fast framework for fine-tuning large language models (LLMs) like LLaMA and Mistral.

> 🚧 This project is currently for learning & testing purposes only. Expect messy code and random experiments!

---

## 🎯 Goals

- ✅ Learn how to fine-tune open-source LLMs with Unsloth
- ✅ Try out QLoRA and LoRA fine-tuning
- ✅ Test different dataset formats
- ✅ Explore model performance with minimal compute

---

## 📆 Setup

Install requirements:

```bash
pip install -U unsloth transformers datasets accelerate
```

GPU (with CUDA) is highly recommended.

---

## 📁 Folder Structure

```bash
.
├── examples/
│   ├── instruction_tuning.py
│   ├── classification_finetune.py
│   └── llama3_qlora_demo.ipynb
├── data/
│   └── sample_dataset.json
└── README.md
```

---

## 🧪 Experiments

| Model       | Dataset         | Notes                      |
|-------------|------------------|-----------------------------|
| LLaMA 2 7B  | Alpaca-style     | Basic instruction tuning   |
| Mistral 7B  | Sentiment sample | Simple classification test |

---

## 📚 References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LoRA Paper (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper (2023)](https://arxiv.org/abs/2305.14314)

---

## 🧑‍💻 Author

Kevin, AI graduate student @ NYCU (currently studying)  
Practicing LLM fine-tuning for fun and learning.


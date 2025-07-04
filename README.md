# LEGAL ASSISTANCE BOT (Based on LLM)

**Author: Tanmay Singh**

---

## Project Overview

This repository presents a **legal question-answering assistant** built on top of Meta’s **LLaMA 3.1 8B Instruct** model and fine-tuned using **PEFT (LoRA)**. It supports natural language queries on legal topics, especially pertaining to the **Indian Penal Code (IPC)**, **Criminal Procedural Code(CRPC)** and **Indian Constitution**.

The assistant can be queried via a Python script or a **Gradio web interface** and is designed to run efficiently on limited GPU memory with `torch.float16`.

---

## Key Features

- Fine-tuned Meta LLaMA 3.1 using PEFT (LoRA) for legal reasoning.
- Instruction-following prompt format (using `<|begin_of_text|>` and `user`/`assistant` role tags).
- Interactive Gradio-based web interface for real-time legal Q&A.
- Compatible with consumer GPUs (via `device_map="auto"` and `torch_dtype=torch.float16`).

> **Tech Stack:** Python, PyTorch, Huggingface Transformers, PEFT, Gradio, CUDA

---

## Setup & Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```


<div align="center">

# 🔧 Transformer Language Model from Scratch

**A hands-on journey into building GPT-style models starting from a simple bigram implementation — step by step, layer by layer.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-frontend-e64c2c.svg)](https://streamlit.io/)

</div>

---

## 📌 Project Overview

The objective of this project was **not merely to replicate GPT-like capabilities**, but to **build a deep understanding** of how transformer-based models work — from first principles.

We **start small**, with a bigram language model, and **incrementally scale up** to a transformer architecture capable of generating coherent sequences.

---

## 🎯 Core Objectives

- 🔨 Build a transformer-based language model from scratch using **PyTorch**
- 🧠 Implement and understand each core component of the transformer:
  - Self-Attention & Multi-Head Attention  
  - Positional Encoding  
  - Feedforward Layers & Layer Normalization
- 📚 Dive into the **mathematical and computational principles** behind transformers
- 🧪 Train the model on small-scale text datasets and test its **generative power**

---

## ⚙️ Technologies & Tools

| Category           | Tools/Libs                                  |
|-------------------|----------------------------------------------|
| Language          | Python                                       |
| Core Libraries    | PyTorch, NumPy                               |
| Notebook Support  | Jupytext (for version-controlled notebooks)  |
| Development Env   | Jupyter Notebooks via VS Code or JupyterLab  |
| UI Interface      | Streamlit                                    |

---

## 🧱 Model Progression

1. **Bigram Language Model** – Simplest form of next-token prediction  
2. **Self-Attention Mechanism** – Capturing token relationships  
3. **Multi-Head Attention** – Parallel attention for richer representations  
4. **Transformer Blocks** – Combining attention, MLP, normalization  
5. **Stacked Architecture** – More depth, better learning  
6. **Text Generation** – Evaluate the output on given prompts

---

## 🖥️ Streamlit App

You can interact with the trained character-level transformer model using a lightweight Streamlit interface.

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run sgpt.py

# 🧠 H-Neurons Analysis on TinyLlama

## 📌 Overview

This repository presents a reproduction and extension of the **H-Neurons analysis** applied to a smaller language model (**TinyLlama**).

The goal is to investigate how specific neurons inside transformer-based language models contribute to:

* factual knowledge retrieval
* domain-specific reasoning
* model interpretability

This work is inspired by research on neuron-level interpretability in large language models.

---

## 🎯 Objectives

* Reproduce the H-Neurons methodology on a lightweight model
* Identify neurons associated with specific knowledge domains
* Evaluate their impact using metrics such as AUROC
* Compare results across multiple datasets

---

## 📊 Datasets

The experiments rely on multiple QA and domain-specific benchmarks:

* **TriviaQA** – general knowledge questions
* **BioASQ** – biomedical question answering
* (Optional) Additional datasets depending on experiments

---

## ⚙️ Methodology

The notebook follows these main steps:

1. **Model Loading**

   * Load TinyLlama or equivalent transformer model

2. **Neuron Activation Extraction**

   * Capture internal activations from hidden layers

3. **H-Neuron Identification**

   * Rank neurons based on their contribution to correct answers

4. **Evaluation**

   * Use **AUROC (Area Under the ROC Curve)** to measure:

     * how well neurons distinguish correct vs incorrect predictions
   * Compare against **random neuron baselines**

---

## 📈 Results

Key findings include:

* Certain neurons consistently encode domain-specific knowledge
* H-Neurons significantly outperform random neuron selection
* Performance varies across datasets (e.g., TriviaQA vs BioASQ)

---

## 🧪 Notebook

The main implementation is available here:

📓 `Amani_M_H_Neurons_GitHub.ipynb`

It contains:

* full pipeline
* experiments
* visualizations
* evaluation metrics

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook
```

---

## 🛠️ Requirements

Typical dependencies include:

* Python 3.10+
* PyTorch
* Transformers (Hugging Face)
* NumPy / Pandas
* Matplotlib / Seaborn

---

## 📚 References

* Research on neuron interpretability in LLMs
* H-Neurons methodology (original paper)
* Transformer-based language models

---

## 👩‍💻 Author

**Amani Mechergui**

* Data Scientist / AI Engineer
* Background in Applied Mathematics & Machine Learning

---

## 📌 Notes

* Notebook cleaned for GitHub rendering (no widget metadata issues)
* Designed for reproducibility and experimentation

---

## ⭐ Contributions

Feel free to:

* open issues
* suggest improvements
* extend experiments to other models

---

# 🧠 H-Neurons Reproduction on TinyLlama
 
> *Reproducing and extending "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" (Gao et al., 2025) on a lightweight, publicly accessible model.*
 
---
 
## 📌 Overview
 
This repository presents a full reproduction of the **H-Neurons** methodology applied to **TinyLlama-1.1B-Chat**, a compact yet architecturally faithful stand-in for the LLaMA-2 family used in the original paper.
 
The core question: **can individual neurons inside a transformer be causally linked to hallucination?**
 
The H-Neurons framework says yes. This notebook:
- implements the full identification pipeline from scratch,
- adapts the paper's GPT-4o-dependent annotation step to a rule-based alternative,
- and runs causal intervention experiments to verify the neurons' effect at inference time.
 
---
 
## 📄 Reference Paper
 
> **H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs**  
> Gao et al., 2025 — [arXiv:2512.01797](https://arxiv.org/abs/2512.01797)  
> Official implementation: [github.com/thunlp/H-Neurons](https://github.com/thunlp/H-Neurons)
 
---
 
## 🎯 Objectives
 
- Reproduce the H-Neurons identification pipeline on a model runnable on a single consumer GPU (Colab T4)
- Implement the **CETT** (Contribution Estimation via Token Transfer) feature extraction method
- Identify neurons associated with hallucinated versus faithful responses using a **sparse L1 logistic classifier**
- Evaluate neuron quality with **AUROC** and compare to random neuron baselines
- Perform **activation intervention experiments** (scaling/ablating H-Neurons) to test causal impact
 
---
 
## 📊 Datasets
 
| Dataset | Task | Split used |
|---|---|---|
| **TriviaQA** (`rc.nocontext`) | Open-domain factual QA — no context passage provided | `train` (first 80 questions) |
| **BioASQ** | Biomedical question answering | Optional extension |
 
TriviaQA's no-context split is specifically chosen because the model must rely entirely on parametric (memorised) knowledge — the setting where hallucination is most likely to emerge.
 
---
 
## ⚙️ Methodology
 
The notebook implements the paper's pipeline across 16 numbered sections:
 
### 1. Model Loading
TinyLlama-1.1B-Chat is loaded in `float16` with automatic device mapping. Key architectural constants (`NUM_LAYERS`, `FFN_DIM`, total neuron search space) are extracted at load time.
 
### 2. Response Collection & Consistency Filtering
For each question, **10 stochastic samples** are generated (temperature=1.0, top-p=0.9). A question is labelled:
 
| Label | Criterion |
|---|---|
| `0` — Faithful | All 10 samples correct |
| `1` — Hallucinated | All 10 samples incorrect |
 
Questions with mixed outcomes are discarded. This consistency filter ensures labels reflect the model's actual knowledge state, not generation randomness.
 
### 3. Answer Token Localisation
Correct-answer tokens are located via character-span matching within the generated response, then mapped to token positions using HuggingFace's `BatchEncoding`. This step approximates the paper's GPT-4o annotation at zero API cost (coverage ~10.6% of samples).
 
### 4. CETT Feature Extraction
For each sample, forward-pass hooks capture the post-activation scalars `z_{j,t}` from every MLP block. The CETT contribution of neuron `j` at token `t` is:
 
$$\text{CETT}_{j,t} = \frac{\|z_{j,t} \cdot W_{\text{down}}[:,j]\|}{\|\text{hidden}_t\|}$$
 
Two aggregate features are extracted per sample:
- `cett_ans` — mean CETT over answer-position tokens
- `cett_other` — mean CETT over non-answer tokens
 
### 5. 3-vs-1 Training Matrix
The classifier is trained with one positive class and three negative classes to isolate neurons specific to *hallucinated* answer generation:
 
| Class | Description |
|---|---|
| **Positive** | Hallucinated answer tokens |
| **Negative 1** | Faithful answer tokens |
| **Negative 2** | Non-answer tokens in hallucinated responses |
| **Negative 3** | Non-answer tokens in faithful responses |
 
### 6. Sparse L1 Logistic Regression
An L1-regularised logistic regression is fitted over the ~124,000-dimensional feature space. L1 penalty drives most weights to exactly zero, producing a sparse set of "H-Neurons" (target sparsity: < 0.1% of total neurons). Regularisation strength `C` is selected by grid search over `{2.0, 1.0, 0.5, 0.1, 0.05, 0.01}`.
 
### 7. Activation Intervention
H-Neurons are causally validated by scaling their `down_proj` input at inference time by factor α:
 
| α | Effect |
|---|---|
| `0.0` | Full ablation → expected fewer hallucinations |
| `0.5` | Partial suppression |
| `1.0` | Baseline (no change) |
| `2.0–3.0` | Amplification → expected more hallucinations |
 
---
 
## 📈 Key Results
 
| Metric | Value |
|---|---|
| Total neurons searched | ~124,000 (22 layers × 5,632 FFN dim) |
| H-Neurons identified | < 0.1% of total (sparse) |
| Classifier AUC | Reported in notebook output |
| Layer distribution | H-Neurons cluster in middle-to-late layers |
| Answer-token coverage | ~10.6% (rule-based vs GPT-4o baseline) |
 
**Takeaways:**
- H-Neurons cluster in **middle-to-late transformer layers**, consistent with the paper's hypothesis that hallucinations emerge during the retrieval-and-composition phase.
- L1 sparsity successfully isolates a very small neuron subset from the ~124K search space.
- The low answer-token coverage (~10.6%) and limited sample size (84 training samples) constrain AUC — a known and discussed limitation of this reproduction.
 
---
 
## 🗂️ Notebook Structure
 
`Amani_M_H_Neurons_GitHub.ipynb` — 33 cells, fully annotated
 
| Section | Description |
|---|---|
| 1 | Environment setup & seed fixing |
| 2 | Model & tokenizer loading (TinyLlama-1.1B-Chat) |
| 3 | TriviaQA dataset loading |
| 4 | Prompt formatting & correctness checking |
| 5 | Stochastic response collection |
| 6 | Answer token localisation |
| 7 | Train/test split (stratified 70/30) |
| 8 | CETT feature extraction (per-token) |
| 9 | Feature extraction loop |
| 10 | 3-vs-1 training matrix assembly |
| 11 | L1 logistic regression + grid search |
| 12 | H-Neuron identification |
| 13 | Layer distribution visualisation |
| 14 | CETT contribution profile plots |
| 15 | Activation intervention experiments |
| 16 | Final summary & discussion |
 
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
jupyter notebook Amani_M_H_Neurons_GitHub.ipynb
```
 
> **Recommended runtime:** Google Colab with a T4 GPU. Expected wall-clock time for the CETT extraction loop: 2–5 minutes at TinyLlama scale.
 
---
 
## 🛠️ Requirements
 
```
python >= 3.10
torch
transformers
accelerate
datasets
scikit-learn
numpy
pandas
matplotlib
seaborn
```
 
Install all at once:
```bash
pip install transformers accelerate datasets scikit-learn numpy pandas matplotlib seaborn
```
 
---
 
## ⚖️ Paper vs. Reproduction: Key Differences
 
| Aspect | Gao et al. (2025) | This Notebook |
|---|---|---|
| Base model | LLaMA-2-7B / 13B | TinyLlama-1.1B-Chat |
| Answer annotation | GPT-4o | Rule-based character matching |
| Answer-token coverage | ~100% | ~10.6% |
| Dataset scale | Full benchmark | 80 questions (first `MAX_QA`) |
| GPU requirement | Multi-GPU cluster | Single T4 (Colab free tier) |
 
These trade-offs are intentional: the goal is **reproducibility on accessible hardware** while preserving methodological fidelity.
 
---
 
## 📚 References
 
- Gao et al. (2025). *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs.* [arXiv:2512.01797](https://arxiv.org/abs/2512.01797)
- Zhang et al. (2024). *CETT: Contribution Estimation via Token Transfer.*
- TinyLlama: [github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- TriviaQA: [nlp.cs.washington.edu/triviaqa](https://nlp.cs.washington.edu/triviaqa/)
- BioASQ: [bioasq.org](http://bioasq.org/)
 
---
 
## 👩‍💻 Author
 
**Amani Mechergui**  
Data Scientist / AI Engineer — Applied Mathematics & Machine Learning
 
---
 
## ⭐ Contributing
 
Contributions are welcome. Ideas for extension:
 
- Replace rule-based annotation with a smaller open-source LLM to improve answer-token coverage
- Scale to more QA pairs or add BioASQ experiments
- Test on other TinyLlama variants or Mistral-7B
- Compare H-Neuron overlap across model families
 
Feel free to open issues or pull requests.
 

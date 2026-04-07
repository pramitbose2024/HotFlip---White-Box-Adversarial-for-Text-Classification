

# HotFlip: White-Box Adversarial Attacks for Text Classification (DistilBERT)

**Duration:** January 2025 – May 2025
**Tech Stack:** Python, PyTorch, Hugging Face Transformers, DistilBERT, HotFlip, SMS Spam Dataset

---

## 📌 Project Overview

This project reimplements and modernizes the 2017 research paper:

**“HotFlip: White-Box Adversarial Examples for Text Classification”**
by Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou.

The original HotFlip algorithm was adapted to a **DistilBERT-based architecture**, enabling adversarial testing on a modern transformer model for spam detection.

The objective was not only to build a high-accuracy classifier but also to evaluate and analyze its robustness under white-box adversarial attacks.

---

## 🎯 Objectives

* Reimplement the **HotFlip white-box adversarial attack**
* Train a **DistilBERT-based spam classifier**
* Generate targeted token-level adversarial examples
* Identify vulnerable tokens influencing model decisions
* Analyze misclassification patterns to improve robustness

---

## 🧠 Model Architecture

* **Base Model:** `distilbert-base-uncased`
* **Task:** Binary Text Classification (Spam vs Ham)
* **Dataset:** SMS Spam dataset
* **Framework:** PyTorch + Hugging Face Transformers
* **Optimizer:** AdamW
* **Loss Function:** CrossEntropyLoss

---

## 🚀 Results

* ✅ Achieved **92% test accuracy** on spam detection
* ✅ Successfully generated adversarial token-level attacks
* ✅ Identified high-gradient influential tokens
* ✅ Demonstrated model vulnerability under white-box perturbations

---

## 🔬 What is HotFlip?

HotFlip is a **white-box adversarial attack** that:

1. Computes gradients of the loss with respect to input embeddings
2. Identifies the most influential token
3. Replaces that token to maximize model loss
4. Causes misclassification with minimal perturbation

Unlike black-box attacks, HotFlip has full access to model gradients, making it a powerful robustness testing method.

---

## ⚙️ Implementation Details

### 1️⃣ Data Processing

* Loaded SMS Spam dataset
* Converted labels into binary format
* Train-test split (80-20)
* Tokenization using `DistilBertTokenizer`

### 2️⃣ Model Training

* Fine-tuned DistilBERT for sequence classification
* Batch size: 16
* Learning rate: 5e-5
* GPU support enabled

### 3️⃣ Gradient-Based Token Importance

* Extracted input embeddings
* Computed gradient of loss w.r.t embeddings
* Identified highest gradient magnitude token

### 4️⃣ Adversarial Token Replacement

* Replaced the most influential token
* Generated perturbed text
* Observed classification changes

---

## 💡 Example Attack

**Original Text:**

> Congratulations! You have won a free lottery ticket.

**Adversarially Perturbed Text:**

> Congratulations! You have won a car lottery ticket.

Even minimal token-level changes can alter model confidence and predictions, exposing robustness gaps.

---

## 📊 Key Learnings

* High accuracy does **not** guarantee robustness.
* Transformer models remain vulnerable to token-level attacks.
* Gradient-based interpretability provides insight into decision boundaries.
* Adversarial testing is essential before deploying NLP systems in real-world applications.

---

## 🏗️ Future Improvements

* Implement full gradient-based optimal token substitution (instead of fixed replacement)
* Add multi-step adversarial attacks
* Improve adversarial training for robustness
* Evaluate robustness on larger datasets
* Compare with other attack methods (TextFooler, DeepWordBug)



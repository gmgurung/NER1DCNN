# Named Entity Recognition (NER) Benchmark
### Evaluating 1D CNN, Simple RNN, and Multi-Layer Perceptron (MLP) Architectures on Sequence Tagging in TensorFlow/Keras
---
## 📌 Project Overview
This project presents a comparative study of three neural network architectures for **Named Entity Recognition (NER)** (sequence tagging) tasks. Using tokenized sentence records (`ner.csv`), the project maps input word sequences to their respective Part-of-Speech (POS) tags and IOB (Inside-Outside-Beginning) entity classifications (e.g., `B-per`, `B-geo`, `I-geo`, `O`). 
Three models are trained, evaluated, and saved:
1. **1D Convolutional Neural Network (1D CNN)**
2. **Simple Recurrent Neural Network (RNN)**
3. **Multi-Layer Perceptron (MLP)**
---
## 📂 Repository Structure
* **`1DCNN.ipynb`**: The complete Jupyter Notebook covering data loading (`ner.csv`), sentence length plotting, train-test splitting (80/20), model definitions, training loops (2 epochs, batch size 64), weight saving, out-of-sample sentence prediction, and performance evaluations.
* **`cnn_model.weights.h5`**: Trained weights for the 1D CNN model.
* **`rnn_model.weights.h5`**: Trained weights for the Simple RNN model.
* **`mlp_model.weights.h5`**: Trained weights for the MLP model.
---
## 🧠 Model Architectures
### 1. 1D CNN Model
Uses temporal convolutions over token embedding sequences to capture localized n-gram context patterns.
* Embedding layer mapping to dense vectors.
* Conv1D layer (128 filters, kernel size 3, ReLU activation) + Dropout (0.3).
* Conv1D layer (64 filters, kernel size 3, ReLU activation) + Dropout (0.3).
* Dense projection (128 units, ReLU) + Softmax classification layer.
### 2. Simple RNN Model
Captures sequential relationships across sentences using recurrent connections.
* Embedding layer mapping to dense vectors.
* SimpleRNN layer (100 hidden units) returning sequence outputs.
* Dense softmax classification layer.
### 3. Multi-Layer Perceptron (MLP) Model
Evaluates each word token independently using time-distributed dense connections.
* Embedding layer mapping to dense vectors.
* TimeDistributed Dense hidden layer (64 units, ReLU).
* TimeDistributed Dense softmax classification layer.
---
## 📈 Evaluation & Results
### 1. Test Set Accuracy Comparison
The models were evaluated on the test partition, yielding the following results:
|
 Model 
|
 Test Accuracy 
|
 Test Loss 
|
|
:---
|
:---:
|
:---:
|
|
**
1D CNN
**
|
**
99.34%
**
|
**
0.0219
**
|
|
**
Simple RNN
**
|
 99.21% 
|
 0.0261 
|
|
**
MLP
**
|
 98.94% 
|
 0.0383 
|
### 2. Entity Tagging Test Case
We tested each model on an out-of-sample sentence:
* **Input Sentence:** *"Bob bought a ticket to NewYork City yesterday."*
* **Target Tags:** `['B-per', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O']`
#### Prediction Outcomes:
* **1D CNN Prediction:** `['B-per', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O']` — **100% Correct** (successfully identified both "Bob" as a person and "NewYork City" as a geographical location).
* **Simple RNN Prediction:** `['B-per', 'O', 'O', 'O', 'O', 'O', 'I-geo', 'O']` — **Incorrect** (missed "NewYork").
* **MLP Prediction:** `['B-per', 'O', 'O', 'O', 'O', 'O', 'I-geo', 'O']` — **Incorrect** (missed "NewYork").
---
## 🚀 Getting Started
### Prerequisites
Install dependencies:
```bash
pip install numpy pandas tensorflow matplotlib scikit-learn tabulate
```
### Running the Project
1. Open the repository root directory.
2. Ensure `ner.csv` is present in the root.
3. Open and run the `1DCNN.ipynb` notebook cell-by-cell.
4. Trained weights will automatically save to their respective `.weights.h5` files, which can then be re-loaded for real-time predictions.

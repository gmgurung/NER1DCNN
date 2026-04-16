# 1D CNN for NER Project

## Project Description
This project focuses on the development and implementation of a 1D Convolutional Neural Network (CNN) for Named Entity Recognition (NER). The primary aim is to accurately identify and classify entities in text data, enhancing the ability to process and organize information from unstructured data sources.

## Models Compared
In this project, we compare the performance of the 1D CNN model against several other models:
- Bi-LSTM
- Conditional Random Fields (CRF)
- Support Vector Machines (SVM)

The comparison is based on accuracy, precision, recall, and F1-score across various datasets.

## Project Structure
```
NER1DCNN/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── 1d_cnn.py
│   ├── bi_lstm.py
│   ├── crf.py
│   ├── svm.py
│
├── notebooks/
│   ├── exploration.ipynb
│   └── results.ipynb
│
├── requirements.txt
├── README.md
└── main.py
```

## Installation Instructions
To set up this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/gmgurung/NER1DCNN.git
   cd NER1DCNN
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide
To run the model:
1. Prepare your dataset and save it in the `data/raw/` directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Results will be saved in the `results` folder for further analysis.

## Technical Details
### 1D CNN Architecture
This model employs a convolutional architecture specifically designed to process sequential data. Key components include:
- **Convolutional Layers**: Extract local patterns and features from the input sequence.
- **Pooling Layers**: Reduce dimensionality and retain essential features.
- **Fully Connected Layers**: Classify the extracted features into different entity categories.

### Collaborator Information
This project is a collaborative effort with Junkai Ge, contributing towards its development and improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
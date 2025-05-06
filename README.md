# Enzyme Classification with XGBoost - A Comparitive Study on the DGEB Benchmark Dataset

This study compares the performance of traditional machine learning methods against foundational models for enzyme commission (EC) classification, by training a k-mer–based XGBoost model on the EC Classification dataset from the Diverse Genomic Embedding Benchmark (DGEB). Model performance is compared against pretrained foundational models on the DGEB that leverage protein sequence embeddings. While XGBoost offered a more interpretable and computationally efficient model, it consistently underperformed relative to foundational models, highlighting the limitations of traditional machine
learning approaches in this domain. These results support the use of foundational models for EC classification, especially in data-scarce settings.

## Setup Instructions

1. **Install Dependencies**  
   Run the following command to install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run cs90u_project.ipynb**


### Project Report

The project report can be found here - [Project Report](https://drive.google.com/file/d/1SwEb9hyeX_6k-G8lQYDEy0oe_erkJhPX/view?usp=sharing)

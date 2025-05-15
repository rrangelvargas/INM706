## Music Genre Classification 

This repository contains code for exploring, pre-processing, and testing CNN+RNN pipelineS for music genre classification using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download).

### Dataset Setup

1. **Download the GTZAN dataset** from the following link:
   [Download Here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)

2. **Unzip the downloaded file.**

3. **Move the contents inside the unzipped folder** (e.g., the `genres_original/` folder) into the existing `data/` directory in this repository.  
   Do **not** replace or delete the existing `data/` folder — just move the dataset contents into it.

*This structure ensures compatibility with all file paths used in the code.*

### Jupyter Notebooks

#### 1. `EDA & Pre-processing.ipynb`

This notebook covers the full **EDA and preprocessing pipeline**, including:
  
- Visualising Mel spectrograms for representative audio samples  
- Segmenting audio into overlapping 4s chunks  
- Extracting and resizing Mel spectrograms for model input  
- Label encoding and splitting data into train/test sets  
- Converting to PyTorch tensors and validating shapes    

#### 2. `test.ipynb`

This notebook is used for **inference and evaluation** of the final trained model:

- Load and test the final model on **new or modern audio samples** outside the GTZAN dataset  
- Evaluate **generalisation** to new musical styles or eras  
- Check how confidently and accurately the model performs on unfamiliar data  




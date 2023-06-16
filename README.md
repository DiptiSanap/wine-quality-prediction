# Wine Quality Prediction using Machine Learning

This repository contains code and resources for predicting the quality of red wine using machine learning. The project is based on the Red Wine Quality dataset obtained from Kaggle.

## Dataset

The dataset used in this project is the [Red Wine Quality dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) from Kaggle. It consists of various chemical properties of red wine along with quality ratings provided by human experts. The dataset contains 1,599 samples with 11 input features and a target variable representing the wine quality. 

The dataset is included in the repository under the `data` directory. It is provided in CSV format (`winequality-red.csv`).

## Prerequisites

To run the code in this repository, you need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook (optional, if you want to run the notebooks)
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- XGBoost

You can install these dependencies using pip:

```
pip install scikit-learn numpy pandas matplotlib seaborn xgboost
```

## Usage

The repository contains the following files and directories:

- `data`: Directory containing the Red Wine Quality dataset.
- `notebooks`: Directory containing Jupyter Notebook files with code and explanations.
- `wine_quality_prediction.ipynb`: Python script that demonstrates the wine quality prediction using machine learning.
- `README.md`: This file.

To run the wine quality prediction using machine learning, you can execute the `wine_quality.ipynb` script. Make sure you have the required dependencies installed. You can modify the script to experiment with different algorithms, data preprocessing techniques, or feature selection methods.

If you prefer a more interactive and explanatory approach, you can explore the Jupyter Notebook files in the `notebooks` directory. Each notebook covers a specific aspect of the project and provides detailed explanations along with code.

## Conclusion
![image](https://github.com/DiptiSanap/wine-quality-prediction/assets/107847530/34d4816f-1e22-473e-a624-48b19c21fc9d)

After performing 5 different algorithms on the dataset, Random forest gave the best performance with 93% accuracy. So I decide to consider it for further evaluation.

## Acknowledgments

- The Red Wine Quality dataset used in this project is sourced from Kaggle, and the credits go to the original authors.
- The creators and contributors of scikit-learn, NumPy, Pandas, Matplotlib, and Seaborn for providing the essential tools for machine learning and data analysis.

# Autism Spectrum Disorder (ASD) Early Detection using Machine Learning

## Overview

This project implements a machine learning framework for the early detection of Autism Spectrum Disorder (ASD), as detailed in the research paper "A Machine Learning Framework for Detection of Autism Spectrum Disorders".  The primary goal is to provide an effective tool for identifying potential ASD cases by leveraging a variety of machine learning techniques.

## Key Features

* **Data Preprocessing:** Includes robust data preprocessing steps such as missing value imputation, feature encoding, and addressing class imbalance using the Random Over Sampler.
* **Feature Scaling:** Employs four different feature scaling methods—Quantile Transformer (QT), Power Transformer (PT), Normalizer, and Max Abs Scaler (MAS)—to optimize the performance of machine learning models.
* **Machine Learning Classification:** Implements eight machine learning classifiers: AdaBoost (AB), Random Forest (RF), Decision Tree (DT), K-Nearest Neighbors (KNN), Gaussian Naïve Bayes (GNB), Logistic Regression (LR), Support Vector Machine (SVM), and Linear Discriminant Analysis (LDA).
* **Performance Evaluation:** Evaluates the classification performance using a comprehensive set of metrics, including accuracy, ROC curve, F1-score, precision, recall, Matthews Correlation Coefficient (MCC), Kappa score, and log loss.
* **Feature Importance Analysis:** Calculates ASD risk factors and ranks the most important attributes using four feature selection techniques: Info Gain Attribute Evaluator (IGAE), Gain Ratio Attribute Evaluator (GRAE), Relief F Attribute Evaluator (RFAE), and Correlation Attribute Evaluator (CAE).

## Requirements

* Python 3.x
* Jupyter Notebook
* Libraries:  (See `requirements.txt` for specific versions)
    * pandas
    * numpy
    * scikit-learn
    * matplotlib

## Usage

1.  **Notebook:** Open the `autism_detection.ipynb` notebook using Jupyter Notebook:

    ```bash
    jupyter notebook notebooks/autism_detection.ipynb
    ```

2.  **Run the Notebook:** Execute the cells in the notebook sequentially to perform data preprocessing, feature scaling, model training, evaluation, and feature importance analysis.  The notebook provides detailed explanations and visualizations of each step.

## Results

The research paper highlights the effectiveness of the proposed framework for early ASD detection. Key findings include:

* Decision Tree Accuracy: 100.00%
* Random Forest Accuracy: 100.00%
* SVM Accuracy: 99.38%
* KNN Accuracy: 98.12%
* Naive Bayes Accuracy: 98.75%
* Logistic Regression Accuracy: 100.00%
* AdaBoost Accuracy: 100.00%
* LDA Accuracy: 80.00%

These results demonstrate high accuracy across several machine learning models, indicating the potential of the framework for ASD detection.  The paper further details the feature importance analysis, identifying key risk factors that can aid healthcare practitioners in diagnosis.

## Contributions

Contributions to this project are welcome.  Feel free to submit pull requests for bug fixes, improvements, or new features.  Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write clear and concise commit messages.
4.  Submit a pull request with a detailed description of your changes.

## Acknowledgments

* The authors of the research paper "A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders" for providing the methodological foundation for this project.
* MIET college, Meerut
* MD. Shahid Sir, MIET

## Citation

If you use this code in your research, please cite the following paper:

> S. M. Mahedy Hasan, MD Palash Uddin, MD Al Mamun, Muhammad Imran Sharif, Anwaar Ulhaq, and Govind Krishnamoorthy, “A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders,” in *IEEE Access*, vol. 11, pp. 15038-15057, 2023, doi: 10.1109/ACCESS.2022.3232490.

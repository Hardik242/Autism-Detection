# Autism-Detection
Using machine learning algorithms to analyze various data points for the early and accurate identification of potential autism spectrum disorder
# Autism Spectrum Disorder (ASD) Early Detection using Machine Learning

## Overview

This project implements a machine learning framework for the early detection of Autism Spectrum Disorder (ASD), as detailed in the research paper "A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders" by Hasan et al. (2023).  The primary goal is to provide an effective tool for identifying potential ASD cases across different age groups (toddlers, children, adolescents, and adults) by leveraging a variety of machine learning techniques.

## Key Features

* **Comprehensive ASD Analysis:** The framework utilizes four standard ASD datasets (Toddlers, Children, Adolescents, and Adults) to ensure broad applicability.
* **Data Preprocessing:** Includes robust data preprocessing steps such as missing value imputation, feature encoding, and addressing class imbalance using the Random Over Sampler.
* **Feature Scaling:** Employs four different feature scaling methods—Quantile Transformer (QT), Power Transformer (PT), Normalizer, and Max Abs Scaler (MAS)—to optimize the performance of machine learning models.
* **Machine Learning Classification:** Implements eight machine learning classifiers: AdaBoost (AB), Random Forest (RF), Decision Tree (DT), K-Nearest Neighbors (KNN), Gaussian Naïve Bayes (GNB), Logistic Regression (LR), Support Vector Machine (SVM), and Linear Discriminant Analysis (LDA).
* **Performance Evaluation:** Evaluates the classification performance using a comprehensive set of metrics, including accuracy, ROC curve, F1-score, precision, recall, Matthews Correlation Coefficient (MCC), Kappa score, and log loss.
* **Feature Importance Analysis:** Calculates ASD risk factors and ranks the most important attributes using four feature selection techniques: Info Gain Attribute Evaluator (IGAE), Gain Ratio Attribute Evaluator (GRAE), Relief F Attribute Evaluator (RFAE), and Correlation Attribute Evaluator (CAE).

## Project Structure

├── data/│   ├── toddlers.csv│   ├── children.csv│   ├── adolescents.csv│   └── adults.csv├── notebooks/│   ├── asd_detection_analysis.ipynb  # Jupyter Notebook with the main analysis├── src/│   ├── preprocessing.py        # Data preprocessing functions│   ├── feature_scaling.py      # Feature scaling methods│   ├── classification.py       # Machine learning classification models│   ├── feature_selection.py    # Feature selection and importance analysis│   ├── evaluation.py         # Evaluation metrics and visualization├── README.md└── requirements.txt
## Requirements

* Python 3.x
* Jupyter Notebook
* Libraries:  (See `requirements.txt` for specific versions)
    * pandas
    * numpy
    * scikit-learn
    * matplotlib
    * seaborn

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Data:** Ensure the ASD datasets (`toddlers.csv`, `children.csv`, `adolescents.csv`, and `adults.csv`) are located in the `data/` directory.
2.  **Notebook:** Open the `asd_detection_analysis.ipynb` notebook using Jupyter Notebook:

    ```bash
    jupyter notebook notebooks/asd_detection_analysis.ipynb
    ```

3.  **Run the Notebook:** Execute the cells in the notebook sequentially to perform data preprocessing, feature scaling, model training, evaluation, and feature importance analysis.  The notebook provides detailed explanations and visualizations of each step.

## Results

The research paper highlights the effectiveness of the proposed framework for early ASD detection.  Key findings include:

* AdaBoost (AB) achieved the highest accuracy for the Toddlers (99.25%) and Children (97.95%) datasets when scaled with the Normalizer.
* Linear Discriminant Analysis (LDA) demonstrated the best performance for the Adolescents (97.12%) and Adults (99.03%) datasets when scaled with the Quantile Transformer (QT).
* The feature importance analysis identifies the most relevant risk factors for ASD, which can aid healthcare practitioners in diagnosis.

## Contributions

Contributions to this project are welcome.  Feel free to submit pull requests for bug fixes, improvements, or new features.  Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write clear and concise commit messages.
4.  Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

* The authors of the research paper "A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders" for providing the methodological foundation for this project.
* The University of California-Irvine (UCI) machine learning repository and Kaggle for providing the ASD datasets.
* Regional Australia Mental Health Research and Training Institute, Manna Institute, NSW, Australia, for funding the research under Grant 0000103935.

## Citation

If you use this code in your research, please cite the following paper:

> S. M. Mahedy Hasan, MD Palash Uddin, MD Al Mamun, Muhammad Imran Sharif, Anwaar Ulhaq, and Govind Krishnamoorthy, “A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders,” in *IEEE Access*, vol. 11, pp. 15038-15057, 2023, doi: 10.1109/ACCESS.2022.3232490.

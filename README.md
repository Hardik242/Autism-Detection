# Autism Spectrum Disorder (ASD) Early Detection using Machine Learning

## Overview

This project implements a machine learning framework for the early detection of Autism Spectrum Disorder (ASD). It includes both a Jupyter Notebook (`model_training.ipynb`) for model training and a Python script (`autism_gui.py`) for a user-friendly graphical interface to make predictions. The project leverages various machine learning techniques to provide an effective tool for identifying potential ASD cases.

### Project Screenshots

Here are some screenshots of the application:

<div>
    <img src="Screenshots/Screenshot (1).png" alt="Screenshot 1" width="400"/>
    <img src="Screenshots/Screenshot (2).png" alt="Screenshot 2" width="400"/>
    <img src="Screenshots/Screenshot (3).png" alt="Screenshot 3" width="400"/>
    <img src="Screenshots/Screenshot (4).png" alt="Screenshot 4" width="400"/>
    <img src="Screenshots/Screenshot (5).png" alt="Screenshot 5" width="400"/>
</div>

## Ready to Run Executable

For ease of use, this project includes a pre-built executable file (`autism_gui.exe`) located in the `Ready to run EXE` directory. You can directly run this executable on your Windows computer to utilize the application without needing to install Python or any dependencies. Simply navigate to the `Ready to run EXE` folder and double-click `autism_gui.exe` to launch the ASD prediction tool.

## Key Features

-   **Ready to Run Executable:** A pre-built `.exe` file for immediate use on Windows.
-   **Data Preprocessing:** Robust preprocessing in the notebook, including missing value imputation, feature encoding, and scaling.
-   **Machine Learning Classification:** Training and evaluation of eight different machine learning classifiers within the `model_training.ipynb` notebook.
-   **GUI Prediction:** A graphical user interface (`autism_gui.py`) allows users to input data and receive ASD predictions.
-   **Model Persistence:** Trained models, preprocessing objects, and column templates are saved as `.pkl` files for use in the GUI.
-   **Clear Results Visualization:** The GUI provides visual representations of prediction results, including pie charts for distribution and bar charts for model confidence.

## File Structure and Workflow

The project is organized as follows:

.
├── Ready to run EXE/ # Directory containing the pre-built executable.
│ └── autism_gui.exe
├── autism_gui.py # Python script for the graphical user interface.
├── model_training.ipynb # Jupyter Notebook for model training and evaluation.
├── models/ # Directory containing trained model files (.pkl).
├── preprocessor.pkl # Saved ColumnTransformer for data preprocessing.
├── template_columns.pkl # Saved list of input feature names.
├── train_combined_final.csv # Training data.
├── test_final.csv # Testing data.
└── README.md # Documentation.

1.  **Ready to Run Application:**

    -   Navigate to the `Ready to run EXE` directory.
    -   Double-click the `autism_gui.exe` file to launch the application.

2.  **Model Training (`model_training.ipynb`):**

    -   Loads training and testing data (`train_combined_final.csv`, `test_final.csv`).
    -   Performs data preprocessing using `scikit-learn` (encoding, scaling).
    -   Trains eight different machine learning models.
    -   Saves the trained models to the `models/` directory as `.pkl` files (e.g., `AB_model.pkl`, `RF_model.pkl`).
    -   Saves the `ColumnTransformer` (`preprocessor.pkl`) and the list of feature names (`template_columns.pkl`).

3.  **GUI Application (`autism_gui.py`):**

    -   Loads the trained models, preprocessor, and feature names.
    -   Creates a graphical interface using `tkinter` for user input.
    -   Takes user input and preprocesses it using the saved `ColumnTransformer`.
    -   Uses the loaded models to make ASD predictions.
    -   Displays the prediction results in a user-friendly format, including visualizations.

## Requirements

-   For running the Python scripts and the notebook:
    -   Python 3.x
    -   Jupyter Notebook
    -   Libraries: (See `model_training.ipynb` and `autism_gui.py` for specific dependencies)
        -   pandas
        -   numpy
        -   scikit-learn
        -   matplotlib
        -   tkinter
-   For running the pre-built executable:
    -   A Windows operating system.

## Usage

1.  **Running the Executable (Recommended):**

    -   Navigate to the `Ready to run EXE` folder.
    -   Double-click `autism_gui.exe`.

2.  **Model Training:**

    -   Open the `model_training.ipynb` notebook using Jupyter Notebook.

        ```bash
        jupyter notebook model_training.ipynb
        ```

    -   Execute the cells in the notebook sequentially to train the models and generate the necessary `.pkl` files.

3.  **Running the Python Script (if you have Python installed):**

    -   Open your terminal or command prompt.
    -   Navigate to the project directory.
    -   Run the `autism_gui.py` script:

        ```bash
        python autism_gui.py
        ```

## Results

The `model_training.ipynb` notebook provides detailed results for each trained model. These results highlight the effectiveness of the machine learning approach for ASD detection.

## Contributions

Contributions to this project are welcome. Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write clear and concise commit messages.
4.  Submit a pull request with a detailed description of your changes.

## Acknowledgments

-   The authors of the research paper "A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders" for providing the methodological foundation for this project.
-   MIET college, Meerut
-   MD. Shahid Sir (College Professor), MIET
-   Hardik Goel (BTech CSE Student)
-   Manas Kush (BTech CSE Student)
-   Harshita Narang (BTech CSE Student)
-   Harsh Bhardwaj (BTech CSE Student)

## Citation

If you use this code in your research, please cite the following paper:

> S. M. Mahedy Hasan, MD Palash Uddin, MD Al Mamun, Muhammad Imran Sharif, Anwaar Ulhaq, and Govind Krishnamoorthy, “A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders,” in _IEEE Access_, vol. 11, pp. 15038-15057, 2023, doi: 10.1109/ACCESS.2022.3232490.

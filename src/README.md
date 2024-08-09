# Cirrhosis Prediction Using Innovative Analysis


## Introduction

Welcome to our web application that helps predict liver function test (LFT) outcomes from medical images. This app is built using Flask and leverages advanced machine learning models to provide accurate predictions for conditions like Non-Alcoholic Fatty Liver Disease (NAFLD), Albumin-Bilirubin (ALBI) scores, and general liver function.

Here's how it works: Users upload images of their LFT results. The app uses Tesseract OCR to read and extract the text from these images. After extracting the data, the app cleans and processes it, then uses pre-trained machine learning models to make predictions.

Our models were chosen after a rigorous process. We first grouped similar data using clustering methods like KMeans, DBSCAN, and Spectral clustering. Then, we trained six different machine learning models for each group and selected the best-performing one.

The app is designed to be user-friendly and responsive. It handles file uploads in the background, so users can continue interacting with the app without any delays.

## Features

- **User-Friendly Interface**: Easy-to-navigate web interface for uploading and processing medical images.
- **OCR Integration**: Uses Tesseract OCR to extract text from uploaded images, ensuring accurate data extraction from LFT reports.
- **Pre-Trained Models**: Utilizes three specific pre-trained models for predicting:
  - Non-Alcoholic Fatty Liver Disease (NAFLD)
  - Albumin-Bilirubin (ALBI) score
  - General liver function test (LFT) outcomes
- **Advanced Data Processing**: Includes data cleaning, preprocessing, and scaling to ensure high-quality input for the models.
- **Model Selection and Evaluation**: Involves clustering (KMeans, DBSCAN, Spectral) and training multiple machine learning models to select the best-performing one for each prediction type.
- **Asynchronous Processing**: Handles file uploads and processing in the background, providing a responsive user experience.
- **Real-Time Progress Tracking**: Allows users to track the progress of their uploads and predictions in real-time.
- **Accurate Predictions**: Provides reliable predictions based on rigorous model training and evaluation, aiding medical professionals in decision-making.
- **Security and Privacy**: Ensures that uploaded medical images and extracted data are handled securely, maintaining user privacy.


## Installation

### Prerequisites

Make sure you have the following installed on your system:

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:

    ```bash
    git clone https://gitlab.computing.dcu.ie/swamins2/2024-mcm-Cirrhosis-Prediction.git
    cd yourproject
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Follow the instructions on the web interface to use the application.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.


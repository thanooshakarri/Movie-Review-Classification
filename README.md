
This code consists of a text classification model that uses logistic regression algorithm to classify text into two categories. The Featurization class uses the TfidfVectorizer from scikit-learn to convert text into numerical features, and also applies preprocessing steps such as removing HTML tags, lemmatization, and language detection. The Classifier class uses the logistic regression algorithm to train on the features and labels provided, and can predict the labels of new text data.

Files:

├── README.md             <- Introduction of repository
├── requirements.txt      <- Python packages requirement file
├── data                  <- Dataset
|   |____ train.csv       <- Train data
|   |____ test.csv        <- Test data
├── src                   <- Source code
|   |____ Classifier.py   <- Text Classifier, pre-processing and feature extraction.
|____ Model.ipynb         <- model

Usage:

    Install the required libraries listed in requirements.txt by running pip install -r requirements.txt in your terminal.

    Open the Model.ipynb notebook in Jupyter or any compatible platform.

    Run the code cells in the notebook to train and test the text classification model.

    Use the Classifier class to classify new text data by creating an instance of the class and calling the predict method with the new data.

    The test file should contain a column TEXT with the text

# Disaster Response Pipeline Project
This repository contains the code I used to build ETL and NLP/ML pipelines and a Flask web app as a part of Udacity's Data Science Nanodegree.

## Table of contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## <a name="installation"></a> Installation
This code uses the following libraries:
- pandas 
- sys
- sqlalchemy
- sqlite3
- nltk
- joblib
- sklearn
- json
- plotly
- flask

This code should run without issues with Python 3.7.10 and up-to-date packages (as of 15 July 2021).

## <a name="motivation"></a> Project Motivation
The aim of this project was to:
1. Build an ETL pipeline which combines the message strings with their categories.
2. Build an NLP/ML pipeline which builds a model, trains it, evaluates it and saves it.
3. Create a Flask web app which shows visualisations of the training data and allows users to enter a message and its genre to receive its categories, as predicted by the model.

## <a name="files"></a> File Descriptions
Apart from the README file, there are a number of files in the repository, as explained below:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- process_data.py # ETL pipeline

- models
|- train_classifier.py # NLP/ML pipeline

- README.md


## <a name="results"></a> Results
1. In order to prepare the model and files needed and run the web app, run the following commands:
    - To run the ETL pipeline that cleans data and stores it in a database
        `python data/process_data.py disaster_messages.csv disaster_categories.csv data/DisasterResponse.db`
        where disaster_messages.csv and disaster_categories.csv are the file paths and names for two datasets inputted into the model and data/DisasterResponse.db is the desired file path and name for the SQL database.
    - To run the NLP/ML pipeline that trains classifier and saves it
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        where data/DisasterResponse.db is the SQL database saved by the ETL pipeline  and models/classifier.pkl is the desired file path and name for the model object saved as a pickle file.

2. Run the following command to run the web app:
    `python app/run.py`

3. Go to http://0.0.0.0:3001/ to see the web app.

## <a name="licensing"></a> Licensing, Authors, and Acknowledgements
Feel free to use the code here as you would like!

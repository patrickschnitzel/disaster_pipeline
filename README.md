# Disaster Response Pipeline Project
This project part of the Udacity Datascientist Nanodegree. It is based on a data set containing real messages that were sent during disaster events. It includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
The classification is done using a Natural Language Processing pipeline to categorize these events so that one can send the messages to an appropriate disaster relief agency.

### Description
The Project is divided in the following sections

- ETL Pipeline to extract data from source, clean and store it in a database
- Machine Learning Pipeline to train and tune a classification model. A detailed justification for selected model, score and hyperparameter is contained in the separate notebook ML Pipeline Preparation.ipynb.
- Web app that recieves, classifies and displays the classification results of entered messages  

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        or run the VSCode ETL configuratiom
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        or run the VSCode training configuratiom
2. Run the following command in the app's directory to run your web app.
    `python run.py`
    or run the VSCode web app configuratiom

3. Go to http://0.0.0.0:3001/

### Acknowledgements
Gratz to Udacity and Figure Eight for providing the dataset and the challenge :)
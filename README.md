# Disaster Response Pipeline Project

### Instructions:
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

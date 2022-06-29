# Disaster Response Pipeline Project

### Instructions:
This project builds an ETL pipeline to process disaster response data, establishes a machine learning pipeline to categorize received messages, and also delivers the pipelines to the web app.  The work will greatly help to categorize disaster response messages and deliver results to different organizations to improve their emergency response capacity. With the application, people can easily tidy tons of received messages and figure out what kind of help is needed by which group of people.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files in the repository 

* app \
|- template \
| |- go.htlm # classification result page of web app \
| |- master.html # main page of web app \
|- run.py # Flask file that runs app 
* data \
|- disaster_categories.csv # data to process \
|- disaster_messages.csv # data to process \
|- DisasterResponse.db # databased used to store cleaned data \ 
|- process_data.py \
|- process_data.ipynb  # the jupyter script used for data exploratory analysis
* models \
|- train_classifier.py \
|- train_classifier.ipynb  # jupyter script used to build and test ML model step-by-step \
|- disaster_response_model.pkl # saved model \
|- DisasterResponse.db # databased used to store cleaned data \
* README.md



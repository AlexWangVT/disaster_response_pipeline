import sys
import re
import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report



def load_data(database_filepath):
    '''
    load the data from sql database, and split the dataframe into X (independent variable) and Y (response).
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM jwang_disaster_pipeline', engine)
    X = df.message.values
    Y = df.drop(['id', 'message'], axis=1).values

    return X, Y


def tokenize(text):
    '''
    tokenize the text data
    '''

    # replace the characters which are not letters and numbers including punctua
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w, pos='v').lower().strip() for w in tokens if w not in stopwords.words("english")]

    return clean_tokens



def build_model():
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )

    parameters = {
        'clf__estimator__criterion': ['gini'],
        'clf__estimator__max_depth': [4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_test = np.argmax(Y_test, axis = 1)
    report = classification_report(Y_test, Y_pred)
    print(report)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    f = open('disaster_response_model.pkl', 'wb')
    pickle.dump(model, f)
    f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
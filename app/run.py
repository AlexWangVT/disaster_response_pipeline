import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('jwang_disaster_pipeline', engine)

# load model
model = joblib.load("../models/disaster_response_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # default genre visual
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # aggregation results grouping by medical help
    med_help_counts = df.groupby('medicalhelp').count()['message']
    med_help_counts.index = ['non-medical-help-message', 'medical-help-message']
    med_help_names = list(med_help_counts.index)

    # aggregation results grouping by earthquake
    earthquake_counts = df.groupby('earthquake').count()['message']
    earthquake_counts.index = ['non-earthquake-message', 'earthquake-message']
    earthquake_names = list(earthquake_counts.index)    

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=med_help_names,
                    y=med_help_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Medical Help Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "message type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=earthquake_names,
                    y=earthquake_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Earthquake Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "message type"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
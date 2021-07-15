import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

class DirectCategoriser(BaseEstimator, TransformerMixin):
    '''
    DirectCategoriser is a transformer class that inherits from sklearns's
    BaseEstimator and TransformerMixin.
    
    METHODS:
        fit: returns itself.
        transform: evaluates whether the genre is 'direct'.
    '''
    def fit(self, x, y=None):
        '''
        This fit function is needed for it to be used in the Pipeline.
        
        INPUTS:
            x: list, series or DataFrame. The independent variable(s).
            y: list, series or DataFrame. The dependent variable(s).
        '''
        return self
    
    def transform(self, X):
        '''
        This transformer takes a series or list and iterates through it. If an 
        individual item is 'direct', then it will return as True. It is 
        intended to be used on the genre column.
        
        INPUTS:
            X: iterable object. Intended to be the genre column.
        
        RETURNS:
            X_categorised: pandas DataFrame. A DataFrame of Trues and Falses
                describing whether each message's genre is direct.
        '''
        X_categorised = []
        for x in X:
            if x == 'direct':
                X_categorised.append(True)
            else:
                X_categorised.append(False)

        return pd.DataFrame(X_categorised)
    

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('ETL_table', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # count number of messages in each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # count number of messages in each category
    category_names = df.columns[4:]
    category_counts = []
    for column in category_names:
        category_counts.append(df[column].sum())
        
    # count average length of each message by category
    avg_lengths = []
    j = 0
    for column in category_names:
        total_length = 0
        i=0
        while i < df[column].shape[0]:
            if df[column][i] == 1:
                total_length += len(df['message'][i])
            i+=1
        avg_length = total_length / category_counts[j]
        avg_lengths.append(avg_length)
        j+=1
    
    # create visuals
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
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=avg_lengths
                )
            ],

            'layout': {
                'title': 'Average length of messages in each category',
                'yaxis': {
                    'title': "Number of characters"
                },
                'xaxis': {
                    'title': "Category"
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
    genre = request.args.get('genre', '')

    # use model to predict classification for query
    #classification_labels = model.predict(np.array(query, genre).reshape(-1,1,1))[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))
    inputs = {'message': [query], 'genre': [genre]}
    inputs_df = pd.DataFrame.from_dict(inputs)
    
    category_names = df.columns[4:]
    classification_labels = model.predict(inputs_df)
    classification_results = dict(zip(category_names, classification_labels[0]))
    
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
import sys
import pandas as pd
import sqlite3
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# download extra features needed
nltk.download('punkt')
nltk.download('wordnet')


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
    

def load_data(database_filepath):
    '''
    Pulls the data table that was created by the ETL Pipeline. Then splits the
    data into X (independent variables) and Y (dependent variables).
    
    INPUTS:
        database_filepath: str. The name of the database to pull the data from.
    
    RETURNS:
        X: pandas DataFrame. Holds the messages and their type (direct, news
            or social).
        Y: pandas DataFrame. Holds the categories of the messages.
    '''
    # load data from database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM ETL_table", con=conn)
    conn.close()

    # separate out data into X and Y    
    X = df[['message', 'genre']]
    Y = df.drop(labels=['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y


def tokenize(text):
    '''
    Processes text data by tokenising, lemmatising, and normalising it.
    
    INPUTS:
        text: str. The text data to be processed.
    
    RETURNS:
        clean_tokens: list of str. A list of the individual words in the 
            inputted text which have been processed individually.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds the model by specifying different transformations for the text-based
    message column and the categorical genre column before using a Random
    Forest classifier.
    
    RETURNS:
        cv: classifier. The multi-output classifier after being tuned by 
            GridSearchCV.
    '''
    # defining the text feature and the transformations to be performed
    text_feature = 'message'
    text_transformer = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
        ])

    # defining the categorical feature and the transformations to be performed
    categorical_feature = 'genre'
    categorical_transformer = DirectCategoriser()

    # combining the above processing steps into one transformer
    transformer = ColumnTransformer([
        ('text_transform', text_transformer, text_feature),
        ('categorical_transform', categorical_transformer, categorical_feature)
    ])

    # define the pipeline
    pipeline = Pipeline ([
        ('tranformer', transformer),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters to be used in the Grid Search
    # processing times meant the number of parameters had to be limited
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
        }
    
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    Calculates and prints out performance metrics for each category.
    
    INPUTS:
        model: classifier. The model produced by build_model().
        X_test: list. The test set of the independent variables, messages and
            genres.
        Y_test: list. The test set of the dependent variables, the category 
            dummy variables.
    '''
    # get model predictions
    Y_pred = model.predict(X_test)
    category_names = Y_test.columns
    Y_pred = pd.DataFrame(Y_pred, category_names)
    
    # calculate performance metrics
    reports = []
    for column in Y_test.columns:
        reports.append(classification_report(Y_test[column], Y_pred[column], \
                                             zero_division=0))
    
    # print out performance metrics for each category
    i=0
    while i < len(category_names):
        category = category_names[i]
        precision = round(reports[i]['macro avg']['precision'], 3)
        f1_score = round(reports[i]['macro avg']['f1-score'], 3)
        recall = round(reports[i]['macro avg']['recall'], 3)
        print ("category: {}, precision: {}, f1-score: {}, recall: {}" \
               .format(category, precision, f1_score, recall))
        i+=1

def save_model(model, model_filepath):
    '''
    Saves the model as a pickle object.
    
    INPUTS:
        model: classifier. The model as returned from build_model().
        model_filepath: str. The desired filepath and name of the model.
    '''
    joblib.dump(model, model_filepath)


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
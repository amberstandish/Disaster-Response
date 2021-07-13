import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reads in the two datasets, one for the messages and one for their 
    categories, and combines them into one DataFrame.
    
    INPUTS:
        messages_filepath: str. A file path for the messages .csv file.
        categories_filepath: str. A file path for the categories .csv file.
    
    RETURNS:
        df: pandas DataFrame. The dataset produced from merging the two 
            inputted files.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    
    return df


def separate_categories(df):
    '''
    Creates a DataFrame where each category is expanded and given a column. 
    Each row consists of 0s and 1s representing whether the message is in that
    category. This DataFrame is then merged with the original DataFrame.
    
    INPUTS:
        df: pandas DataFrame. The DataFrame produced by load_data().
    
    RETURNS:
        df_expanded: pandas DataFrame. The inputted DataFrame with the 
            categories column expanded.
        category_names: list. Contains all of the names of the categories.
    '''
    # create a dataframe of the 36 individual category columns and merge it 
    # with the id column to faciliate matching later
    categories = df['categories'].str.split(';', expand=True)
    categories = pd.DataFrame(df['id']).reset_index() \
        .merge(categories.reset_index(), on='index')     
    categories = categories.drop(columns = ['index'])
    
    # select the first row of the categories dataframe, excluding the id column
    row = categories.iloc[:1,1:]

    # extract from the row a list of the category names
    extract_function = lambda x: x.split('-')
    rows_list = row.values.tolist()[0]

    category_names = [extract_function(row)[0] for row in rows_list]
    
    # add the name for id to the get the final list of desired column names
    category_colnames = ['id'] + category_names
    
    # rename the columns of the categories dataset
    categories.columns = category_colnames
    
    # convert the values in the category columns to be one or zero and numeric
    for column in categories.iloc[:,1:]:
        # set each value to be the last character of the string
        column_list = categories[column].values.tolist()
        categories[column] = [extract_function(row)[1] for row in column_list]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
   
    # drop the original categories column from df
    df_expanded = df.drop(labels=['categories'], axis=1)
    
    # concatenate the original dataframe with the new categories DataFrame
    df_expanded = df.merge(categories)
    
    return df_expanded, category_names


def clean_data(df, category_names):
    '''
    Drops duplicates in the DataFrame and checks that no duplicates remain.
    
    INPUTS:
        df: pandas DataFrame. The DataFrame returned by separate_categories().
        category_names: list. The names of the categories; returned by
            separate_categories().
    
    RETURNS:
        df_clean: pandas DataFrame. The inputted DataFrame with duplicates
            removed.
    '''
    # drop duplicates
    df_clean = df.drop_duplicates(subset=category_names)
    
    # check duplicates have been removed
    if df_clean.iloc[:,1:].duplicated().sum() == 0:
        print('Duplicates removed successfully.')
    else:
        print('Duplicates are still present.')
    
    return df_clean


def save_data(df, database_filename):
    '''
    Loads the DataFrame to the database.
    
    INPUTS:
        df: pandas DataFrame. The DataFrame returned by clean_data().
        database_filename: str. The name of the database to store df in.
    '''
    engine = create_engine('sqlite:///data/' + database_filename)
    df.to_sql('ETL_table', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df, category_names = separate_categories(df)
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
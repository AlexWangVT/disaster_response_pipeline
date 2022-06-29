from heapq import merge
import sys
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load raw data from messages and categories csv file and merge them
    category_df - Response variable
    message_df - Independent variable
    merge_df - merged dataframe
    '''
    category_df = pd.read_csv(categories_filepath)
    category_df.replace(2, 1)
    message_df = pd.read_csv(messages_filepath)
    merge_df = category_df.merge(message_df, on='id')
    
    return merge_df


def clean_data(df):
    '''
    This function is used for data preprocessing which includes multiple steps:
    1. extract the column of the response variable.
    2. select the first row of the categories dataframe and use this row to extract a list of new column names for categories.
    3. convert category value to numbers 0 or 1
    '''
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # extract a list of new column names for categories
    category_colnames = [re.sub(r"[^a-zA-Z]", "", text) for text in row]
    categories.columns = category_colnames

    # convert category values to numeric data type
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column]).astype('Int64')

    # replace value 2 by 1
    categories = categories.replace(2, 1)

    # Drop the original categories column
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    save the the cleaned data to sql database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('jwang_disaster_pipeline', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
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
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Loads data from csv files and concats this into a DataFrame

    Parameters
    ----------
    messages_filepath
        path to messages csv file which contains messages created during dissasters
    categories_filepath
        path to categories csv file which categorizes these messages

    Returns
    -------
        DataFrame which contains data of ``messages_filepath`` and ``categories_filepath``.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'inner')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1].str.replace(r'-\d', '')

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.values

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns=["categories"], inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    return pd.concat([df,categories], axis="columns")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans DataFrame

    Parameters
    ----------
    df
        DataFrame to clean
    Returns
    -------
        Cleaned DataFrame ``df``
    """
    return df.drop_duplicates()

def save_data(df: pd.DataFrame, database_filename: str):
    """
    Stores dataframe in SQL database
    Parameters
    ----------
    df
        DataFrame to store
    database_filename
        file name of database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponseData', engine, index=False, if_exists='replace')

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
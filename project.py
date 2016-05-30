"""
project.py - run this to re-create my best submission to the Shelter Animals
Kaggle competition. See: github.com/jakesherman/shelter-animals-kaggle
"""

from fuzzywuzzy import fuzz
import json
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn import preprocessing
import wikipedia


"""
================================================================================
    CATEGORIZING DOG BREEDS INTO THE AMERICAN KENNEL CLUB TAXONOMY
================================================================================
"""


def identify_classification(breed, page_content):
    """
    Identifies the classification for a breed in the AKC taxonomy from the 
    content of the Wikipedia page 'List of dog breeds recognized by the 
    American Kennel Club'.
    """
    start, i = page_content.index(breed) + len(breed) + 1, 1
    while '\n' not in page_content[start:start + i + 1][-2:]:
        i += 1
    classification = page_content[start:start + i]
    if classification[0] == ' ':
        return classification[1:]
    else:
        return classification


def create_breed_to_classification_dict():
    """
    Create a dictionary that maps dog breeds to their taxonomic classification
    by the American Kennel Club via the Wikipedia API.
    """
    akc_breeds = wikipedia.page(
        'List of dog breeds recognized by the American Kennel Club')
    exclude = [u'Dog breed', u'American Kennel Club', u'List of dog breeds', 
               u'List of dog breeds recognized by the Canadian Kennel Club']
    breeds = [page for page in akc_breeds.links if page not in exclude]
    page_content = akc_breeds.content
    classifications = {}
    for breed in breeds:
        
        # Manual fixes, cases where the links and link text are not the same
        breed = (breed
            .replace(' (dog)', '')
            .replace(' (dog breed)', '')
            .replace('American Cocker Spaniel', 'Cocker Spaniel')
            .replace('American Eskimo Dog', 'American Eskimo Dog (Miniature)')
            .replace('Australian Silky Terrier', 'Silky Terrier')
            .replace('Bergamasco Shepherd', 'Bergamasco')
            .replace('English Mastiff', 'Mastiff')
            .replace('Griffon Bruxellois', 'Brussels Griffon')
            .replace('Hungarian Vizsla', 'Vizsla')
            .replace('Rough Collie', 'Collie'))
        try:
            classifications[breed] = identify_classification(breed, 
                page_content)
        except:
            print 'COULD NOT CLASSIFY THE FOLLOWING BREED:', breed
    return classifications


def map_breeds_to_taxonomy(train, test):
    """
    Use fuzzy string matching (~ Levenshtein distance) to match dog breeds from
    the AKC list of dog breeds Wikipedia page to the Breed variable in the 
    training and test sets. Output a dictionary of training/test breed to the 
    set of AKC groups for that breed (as many of the dogs are mixed breed, this 
    set may have more than one value).
    """
    if os.path.isfile('dogbreeds.json'):
        with open('dogbreeds.json') as data_file:    
            dogbreeds = json.load(data_file)
        return dogbreeds
    train_breeds = list(train['Breed'].unique())
    test_breeds = list(test['Breed'].unique())
    all_breeds = list(set(train_breeds + test_breeds))
    breed_classifications = {}
    for train_breed in all_breeds:
        classes = []
        
        # Remove the word 'Mix' and identify breeds separately for strings 
        # separated by a /, since this often distinguishes between 2 diff breeds
        train_breed_clean = train_breed.replace(' Mix', '')
        train_breed_split = train_breed_clean.split('/')
        for partial_breed in train_breed_split:
            high_score, current_class = 0, None
            for classified_breed in classifications.keys():
                score = fuzz.token_sort_ratio(partial_breed, classified_breed)
                if score > high_score:
                    high_score = score
                    current_class = classifications[classified_breed]
            classes.append(current_class)
            
        # Split cases where the breed name is 'A & B' separately into A, B
        for myclass in classes:
            if '&' in myclass:
                for subclass in myclass.split(' & '):
                    classes.append(subclass)
                classes = [c for c in classes if c != myclass]
        breed_classifications[train_breed] = set(classes)
    dogbreeds = dict([(k, list(v)) for k, v in breed_classifications.items()])
    with open('dogbreeds.json', 'w') as f:
         json.dump(dogbreeds, f)
    return map_breeds_to_taxonomy(train, test)


"""
================================================================================
    FEATURE ENGINEERING
================================================================================
"""


def create_outcomes_drop_cols(train, test):
    """
    Delete unneeded columns, and split the outcomes into separate numpy arrays.
    """
    le = preprocessing.LabelEncoder()
    le.fit(train['OutcomeType'])
    outcomes = le.transform(train['OutcomeType'])
    train = train.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype'], axis = 1)
    test = test.drop(['ID'], axis = 1)
    return train, test, outcomes, le


def create_sex_variables(data):
    """
    Create Sex and Neutered features from SexuponOutcome, which was really 
    two features in one - gender and neutered/intact. 
    """
    SexuponOutcome = data['SexuponOutcome'].fillna('Unknown')
    results = []
    for row in SexuponOutcome:
        row = row.split(' ')
        if len(row) == 1:
            row = ['Unknown', 'Unknown']
        results.append(row)
    NeuteredSprayed, Sex = zip(
        *[['Neutered', x[1]] if x[0] == 'Spayed' else x for x in results])
    return (data.assign(Neutered = NeuteredSprayed).assign(Sex = Sex)
        .drop(['SexuponOutcome'], axis = 1))


def create_age_in_years(data):
    ages = list(data['AgeuponOutcome'].fillna('NA'))
    results = []
    units = {'days': 365.0, 'weeks': 52.0, 'months': 12.0}
    for age in ages:
        if age == 'NA':
            results.append('NA')
        else:
            duration, unit = age.split(' ')
            results.append(float(duration) / units.get(unit, 1.0))
    impute = np.mean([age for age in results if age != 'NA'])
    ages = [age if age != 'NA' else impute for age in results]
    return (data
            .assign(Age = preprocessing.scale(ages))
            .drop(['AgeuponOutcome'], axis = 1))


def time_of_day(hour):
    if hour > 4 and hour < 12:
        return 'morning'
    elif hour >= 12 and hour < 18:
        return 'afternoon'
    else:
        return 'evening/night'
    

def day_of_the_week(DateTime):
    return datetime.datetime.strptime(DateTime, '%Y-%m-%d %H:%M:%S').weekday()


def create_date_variables(data):
    return (data
        .assign(Year = data.DateTime.map(lambda x: x[:4]))
        .assign(Month = data.DateTime.map(lambda x: x[5:7]))
        .assign(Day = data.DateTime.map(lambda x: day_of_the_week(x)))
        .assign(TimeOfDay = data.DateTime.map(
            lambda x: time_of_day(int(x[11:13]))))
        .drop(['DateTime'], axis = 1))


def assign_dog_breeds(data, dogbreeds):
    """
    Create binary indicator variables for the AKC dog classifications.
    """
    unique_breeds = set(
        [breed for breeds in dogbreeds.values() for breed in breeds])
    breed_to_position = dict([(x, i) for i, x in enumerate(unique_breeds)])
    vectors = []
    for breed, animal in data[['Breed', 'AnimalType']].values.tolist():
        vector = [0] * len(unique_breeds)
        if animal == 'Dog':
            breed = dogbreeds[breed]
            for subbreed in breed:
                vector[breed_to_position[subbreed]] += 1
        vectors.append(vector)
    columns = ['AKC_Class_' + x[1] for x in sorted(
        [(v, k) for k, v in breed_to_position.items()])]
    dogbreeds_df = pd.DataFrame(vectors, columns = columns)
    return pd.concat([data, dogbreeds_df], axis = 1)


def create_mix(data):
    """
    For both cats and dogs, is the animal a mix of multiple breeds? We can 
    determine this both by searching for the string 'Mix' in the name, AND by 
    looking to see if dogs have been classified into 2 or more AKC groups.
    """
    dogbreedcols = [col for col in list(data) if 'AKC_Class_' in col]
    data['NumDogBreeds'] = data[dogbreedcols].sum(axis = 1)
    data['Mix'] = data['Breed'].map(lambda x: x.find('Mix'))
    data['Mix'] = np.where(data['Mix'] > 0, 1, 
        np.where(data['NumDogBreeds'] > 1, 1, 0))
    return data


def create_hair_length(data):
    """
    For cats, creates binary indicator variables for whether the cat has long,
    medium, or short hair.
    """
    data['ShortHair'] = np.where(data['Breed'].map(
        lambda x: x.find('Short')) > 1, np.where(
        data['AnimalType'] == 'Cat', 1, 0), 0)
    data['MediumHair'] = np.where(data['Breed'].map(
        lambda x: x.find('Medium')) > 1, np.where(
        data['AnimalType'] == 'Cat', 1, 0), 0)
    data['LongHair'] = np.where(data['Breed'].map(
        lambda x: x.find('Long')) > 1, np.where(
        data['AnimalType'] == 'Cat', 1, 0), 0)
    return data


def create_breed_variables(data, dogbreeds):
    return (data
        .pipe(assign_dog_breeds, dogbreeds = dogbreeds)
        .pipe(create_mix)
        .pipe(create_hair_length)
        .drop(['Breed'], axis = 1))


def extract_unique_colors(train):
    """
    Extract a set of unique colors from the training set for colors with 
    30 or more animals.
    """
    colors = {}
    for color in list(train['Color']):
        color = re.split('\W+', color)
        for subcolor in color:
            try:
                colors[subcolor] += 1
            except:
                colors[subcolor] = 1
    return set([color for color, count in colors.items() if count >= 30])


def create_color_variables(data, colors):
    """
    Create binary indicator variables for each color identified in the training
    set by extract_unique_colors().
    """
    vectors = []
    to_position = dict([(c, i) for i, c in enumerate(colors)])
    for color in list(data['Color']):
        vector = [0] * len(colors)
        color = re.split('\W+', color)
        for subcolor in color:
            try:
                vector[to_position[subcolor]] += 1
            except:
                pass
        vectors.append(vector)
    columns = ['Color_' + x[1] for x in sorted(
            [(v, k) for k, v in to_position.items()])]
    colors_df = pd.DataFrame(vectors, columns = columns)
    return pd.concat([data, colors_df], axis = 1).drop(['Color'], axis = 1)


def create_has_name(data):
    data['HasName'] = np.where(data['Name'].isnull(), 0, 1)
    return data.drop(['Name'], axis = 1)


def transform_animal_type(data):
    data['AnimalType'] = np.where(data['AnimalType'] == 'Cat', 1, 0)
    return data


def one_hot_encode(DataFrame, column):
    """
    Replace [column] in [DataFrame] with binary columns for each distinct value
    in [column], each with the name [column]_[value].
    """
    to_col = dict([(n, i) for i, n in enumerate(list(
        DataFrame[column].unique()))])
    mat = np.zeros((len(DataFrame.index), len(to_col)))
    for i, val in enumerate(list(DataFrame[column])):
        mat[i, to_col[val]] += 1
    columns = [column + '_' + str(x[1]) for x in sorted(
            [(v, k) for k, v in to_col.items()])]
    dfs = [DataFrame, pd.DataFrame(mat.astype(int), columns = columns)]
    return pd.concat(dfs, axis = 1).drop([column], axis = 1)


def do_one_hot_encoding(data):
    columns = list(data)[1:7]
    for column in columns:
        data = data.pipe(one_hot_encode, column = column)
    return data


def feature_engineering(data, dogbreeds, colors):
    return (data
         .pipe(create_sex_variables)
         .pipe(create_date_variables)
         .pipe(create_age_in_years)
         .pipe(create_breed_variables, dogbreeds = dogbreeds)
         .pipe(create_color_variables, colors = colors)
         .pipe(create_has_name)
         .pipe(transform_animal_type)
         .pipe(do_one_hot_encoding)
         .pipe(np.array))


def main():

    # Read in the data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Classify the breeds in the train/test set into the AKC taxonomy
    dogbreeds = map_breeds_to_taxonomy(train, test)

    # Feature engineering
    train, test, outcomes, outcomes_le = create_outcomes_drop_cols(train, test)
    colors = extract_unique_colors(train)
    test = feature_engineering(test, dogbreeds, colors)
    train = feature_engineering(test, dogbreeds, colors)

    # Modeling


if __name__ == '__main__':
    main()

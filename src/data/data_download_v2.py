# Version 2 of data ingestion Python script
'''
The following script does the following:
- checks for cloned Panacea Lab Twitter repo; if not present in current directory, clones the Repo
- pulls most recent data from Panacea repo, to ensure contents are up-to-date
- creates list of daily folders located in Panacea repo
- creates raw_dailies & processed_dailies folder within my covid repo, if it does not exist
- makes daily folders in covid_disinfo_detect raw & procssed data repositories
- gets Tweet IDs and convert to txt format without headers & index
- checks for JSON files in raw folders
- hydrates Twitter data based on IDs via twarc library
- uploads JSON to storage bucket, then creates a compressed copy for local storage
- cleans data into pickle format, storing in processed_dailies day folder
'''
import git
import re
from pathlib import Path
import pandas as pd
import time
import subprocess
from google.cloud import storage


def setpath():
    return Path.home()


def clone_panacea_repo(homepath):
    try:
        print('Cloning repository...')
        gitrepo = 'https://github.com/thepanacealab/covid19_twitter.git'
        git.Repo.clone_from(gitrepo, homepath / 'thepanacealab_covid19')
        print('Repo cloned.')
    except Exception as e:
        print(e)
        
        
def panacea_pull(panacearepopath):
    g = git.cmd.Git(panacearepopath)
    result = g.pull()
    return result


def make_raw_folders(myrepopath, daily_list):
    # for day in list of daily folders from Panacea Labs GitHub repo
    for day in daily_list:
        if (myrepopath / 'data' / 'raw_dailies' / day).exists():
            pass
        else:
            newpath = myrepopath / 'data' / 'raw_dailies' / day
            newpath.mkdir()
            
            
def make_proc_folders(myrepopath, daily_list):
    # for day in list of daily folders from Panacea Labs GitHub repo
    for day in daily_list:
        if (myrepopath / 'data' / 'processed_dailies' / day).exists():
            pass
        else:
            newpath = myrepopath / 'data' / 'processed_dailies' / day
            newpath.mkdir()
            
            
def get_txt_data(myrepopath, panacearepopath, daily_list):
    # for day in list of daily folders from Panacea Labs GitHub Repo
    for day in daily_list:
        # create path variables to access data in Panacea repo, and path to local storage folder
        storagepath = myrepopath / 'data' / 'raw_dailies' / day
        datapath = panacearepopath / 'dailies' / day
        # get list of contents within local daily storage folder 
        files = [x.name for x in storagepath.iterdir()]
        # if txt file with that date is in daily storage folder, print confirmation
        if f'{day}_clean-dataset.txt' in files:
            print(f'Txt detected in {storagepath}')
        # else read in compressed tsv file with Tweet IDs from Panacea repo & store txt file
        # with Tweet IDs in local daily storage folder
        else:
            df = pd.read_csv(f'{datapath}/{day}_clean-dataset.tsv.gz',
                             sep='\t', usecols=['tweet_id'], compression='gzip')
            df.to_csv(f'{storagepath}/{day}_clean-dataset.txt', header=None, index=None)
            
            
def main_setup():
    # set up path to current working directory & path to directory containing Panacea data
    homepath = setpath()
    myrepopath = homepath / 'covid_disinfo_detect'
    panacearepopath = homepath / 'thepanacealab_covid19'
    if myrepopath.exists():
        pass
    else:
        myrepopath.mkdir()
    # if Panacea lab folder in working directory, print confirmation, else clone the repo
    if 'thepanacealab_covid19' in [x.name for x in homepath.iterdir()]:
        print('Panacea Labs COVID-19 GitHub has already been cloned...')
    else:
        clone_panacea_repo(path)
        
    # pull any recent updates from Panacea Lab repo
    pull_result = panacea_pull(panacearepopath)
    print(pull_result)
    # create list of daily folders located in Panacea repo (which contains data we need to access)
    file_ignore = ['README.md', '.ipynb_checkpoints']
    daily_list = [x.name for x in sorted((panacearepopath / 'dailies').iterdir())\
                  if x.name not in file_ignore]
    # check to see if data sub-directory exists in my repo
    mydatapath = myrepopath / 'data'
    if mydatapath.exists(): 
        pass
    else:
        mydatapath.mkdir()
    
    # if raw_dailies sub-folder exists make folders for raw data and get text of IDs
    if 'raw_dailies' in list(x.name for x in mydatapath.iterdir()):
        make_raw_folders(myrepopath, daily_list)
        get_txt_data(myrepopath, panacearepopath, daily_list)
    # else make raw_dailies folder, then make folders for raw data and get text of IDs
    else:
        mydailypath = mydatapath / 'raw_dailies'
        mydailypath.mkdir()
        make_raw_folders(myrepopath, daily_list)
        get_txt_data(myrepopath, panacearepopath, daily_list)
        
    # check to see if processed_dailies sub-folder exists then create daily folders    
    if 'processed_dailies' in list(x.name for x in mydatapath.iterdir()):
        make_proc_folders(myrepopath, daily_list)
    else:
        myprocdailypath = mydatapath / 'processed_dailies'
        myprocdailypath.mkdir()
        make_proc_folders(myrepopath, daily_list)
        
        
def blob_exists(bucket_name, source_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    return blob.exists()


def storage_check(daily_list):
    bucket_name = 'thepanacealab_covid19twitter'
    nojson = []
    for day in daily_list:
        source_file_name1 = f'dailies/{day}/{day}_clean-dataset.json'
        source_file_name2 = f'dailies/{day}/panacealab_{day}_clean-dataset.json'
        json1_exist = blob_exists(bucket_name, source_file_name1)
        json2_exist = blob_exists(bucket_name, source_file_name2)
        if json1_exist or json2_exist == True:
            pass
        else:
            nojson.append(day)
    return nojson


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    
    
def load_data(filename, chunksize=10000):
    good_columns = [
        'created_at',
        'entities',
        'favorite_count',
        'full_text',
        'id_str',
        'in_reply_to_screen_name',
        'in_reply_to_status_id_str',
        'is_quote_status',
        'lang',
        'retweet_count',
        'source',
        'user',
        'quoted_status_id_str',
        'quoted_status_permalink'
    ]
    chunks = pd.read_json(
        filename, lines=True, chunksize=chunksize,
        dtype={'id_str': str, 'in_reply_to_status_id_str': str, 'quoted_status_id_str': str},
        compression='gzip'
    )
    df = pd.concat(chunk for chunk in chunks)[good_columns]
    return df


def entity_extraction(entity, component, urls=False, user_mentions=False):
    try:
        if urls == True:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([url['url'] for url in entity[component]])
        elif user_mentions == True:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([mention['screen_name'] for mention in entity[component]])
        else:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([comp['text'] for comp in entity[component]])
    except:
        return None
    
    
def source_extract(text):
    try:
        regex = re.compile(r'(?<=>).*?(?=<)', re.I)
        return regex.search(text).group()
    except AttributeError as e:
        return None
    
    
def quoted_status_extract(status):
    try:
        return status['url']
    except:
        return None
    
    
def clean_panacea_data(dataframe):
    user_components = [
        'created_at', 'description', 'favourites_count', 'followers_count', 'friends_count',
        'id_str', 'location', 'name', 'profile_image_url_https', 'screen_name',
        'statuses_count', 'verified'
    ]
    dataframe['hashtags'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'hashtags'))
    dataframe['symbols'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'symbols'))
    dataframe['urls'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'urls', urls=True))
    dataframe['user_mentions'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'user_mentions', user_mentions=True))
    dataframe['tweet_source'] = dataframe['source'].apply(source_extract)
    for comp in user_components:
        dataframe[f'user_{comp}'] = dataframe['user'].apply(lambda user: user[comp])
    dataframe['quoted_status_url'] = dataframe['quoted_status_permalink'].apply(quoted_status_extract)
    dataframe.drop(labels=['user', 'entities', 'source', 'quoted_status_permalink'], axis=1, inplace=True)
    dataframe.fillna('none', inplace=True)
    return dataframe


def clean_data_wrapper(daypath, myprocdatapath, day):
    print('Loading data...')
    df = load_data(f'{daypath}/{day}_clean-dataset.json.gz')
    print('Cleaning data...')
    df = clean_panacea_data(dataframe=df)
    print(f'Cleaned data, converting data for date {day} to pickle format...')
    df.to_pickle(f'{myprocdatapath}/{day}/{day}_clean-dataset.pkl')
    print(f'Transferred file to following location: {myprocdatapath / day / day}...\n')
    
    
def twarc_gather(myrawdatapath, myprocdatapath, daily_list):
    for day in daily_list:
        daypath = myrawdatapath / day
        twarc_command = f'twarc hydrate {daypath}/{day}_clean-dataset.txt > {daypath}/{day}_clean-dataset.json'
        gzip_command = f'gzip -k {daypath}/{day}_clean-dataset.json'
        try:
            print(f'Hydrating data for {day}...')
            subprocess.call(twarc_command, shell=True)
            print('Done gathering data via twarc, compressing JSON...')
            subprocess.call(gzip_command, shell=True)
            print('File compressed! Now uploading JSON file to Storage Bucket...')
            upload_blob(
                bucket_name='thepanacealab_covid19twitter',
                source_file_name=f'{daypath}/{day}_clean-dataset.json',
                destination_blob_name=f'dailies/{day}/{day}_clean-dataset.json'
            )
            print(f'JSON file uploaded to Storage Bucket, now removing JSON from {day} folder...')
            filepath = daypath / f'{day}_clean-dataset.json'
            # remove JSON file
            filepath.unlink()
            print(f'JSON removed from {day} folder!')
            # clean data
            clean_data_wrapper(daypath, myprocdatapath, day)
        except Exception as e:
            print(e)
            
            
def main_gather():
    # set up path to current working directory & path to directory containing Panacea data
    homepath = setpath()
    myrepopath = homepath / 'covid_disinfo_detect'
    panacearepopath = homepath / 'thepanacealab_covid19'
    myrawdatapath =  myrepopath / 'data' / 'raw_dailies'
    myprocdatapath = myrepopath / 'data' / 'processed_dailies'
    # create list of daily folders located in Panacea repo (which contains data we need to access)
    file_ignore = ['README.md', '.ipynb_checkpoints']
    daily_list = [x.name for x in sorted((panacearepopath / 'dailies').iterdir())\
                  if x.name not in file_ignore]
    # see what daily data we do not have in storage bucket
    nojson = storage_check(daily_list)
    previous3 = nojson[-3:]
    print(f'\nTotal of {len(nojson)} folders do not contain a JSON file:\n{nojson}\n')
    print(f'Gathering data for the previous 3 days without JSONs:\n{previous3[::-1]}')
    twarc_gather(myrawdatapath, myprocdatapath, previous3[::-1])
    
    
def main_program():
    main_setup()
    main_gather()
    
    
if __name__ == '__main__':
    main_program()
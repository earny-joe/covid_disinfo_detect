'''
The following script does the following:
- checks for cloned Panacea Lab Twitter repo; if not present in current
directory, clones the Repo
- pulls most recent data from Panacea repo, to ensure contents are
up-to-date
- creates list of daily folders located in Panacea repo
- creates raw_dailies folder within my covid repo, if it does not exist
- makes daily folders in covid_disinfo_detect raw data repository
- gets Tweet IDs and convert to txt format without headers & index
- checks for JSON files in raw folders
- hydrates Twitter data based on IDs via twarc library
'''
import git
from pathlib import Path
import pandas as pd
import subprocess
from google.cloud import storage


def setpath():
    return Path.cwd()


def clone_panacea_repo(path):
    try:
        print('Cloning repository...')
        gitrepo = 'https://github.com/thepanacealab/covid19_twitter.git'
        git.Repo.clone_from(gitrepo, path / 'thepanacealab_covid19')
        print('Repo cloned.')
    except Exception as e:
        print(e)


def panacea_pull():
    g = git.cmd.Git('thepanacealab_covid19')
    result = g.pull()
    return result


def make_folders(path, daily_list):
    # for day in list of daily folders from Panacea Labs GitHub repo
    for day in daily_list:
        daypath = path/'covid_disinfo_detect'/'data'/'raw_dailies'/day
        if daypath.exists():
            pass
        else:
            newpath = daypath
            newpath.mkdir()


def get_txt_data(path, repopath, daily_list):
    # for day in list of daily folders from Panacea Labs GitHub Repo
    for day in daily_list:
        # create path variables to access data in Panacea repo
        # and path to local storage folder
        storagepath = path/'covid_disinfo_detect'/'data'/'raw_dailies'/day
        datapath = repopath/'dailies'/day
        # get list of contents within local daily storage folder
        files = [x.name for x in storagepath.iterdir()]
        # if txt file with that date is in daily storage folder
        # and print confirmation
        if f'{day}_clean-dataset.txt' in files:
            print(f'Txt detected in {storagepath}')
        # else read compressed tsv file with Tweet IDs from Panacea repo
        # & store txt file with Tweet IDs in local daily storage folder
        else:
            df = pd.read_csv(
                f'{datapath}/{day}_clean-dataset.tsv.gz', sep='\t',
                usecols=['tweet_id'], compression='gzip'
            )
            df.to_csv(
                f'{storagepath}/{day}_clean-dataset.txt',
                header=None, index=None
            )


def main_setup():
    # set up path to current working directory & path to directory
    # containing Panacea data
    path = setpath()
    covidpath = path/'covid_disinfo_detect'
    repopath = path/'thepanacealab_covid19'
    # if Panacea lab folder in working directory
    # print confirmation, else clone the repo
    if path / 'thepanacealab_covid19' in path.iterdir():
        print('Panacea Labs COVID-19 GitHub has already been cloned...')
    else:
        clone_panacea_repo(path)
    # pull any recent updates from Panacea Lab repo
    pull_result = panacea_pull()
    print(pull_result)
    # create list of daily folders located in Panacea repo
    # (which contains data we need to access)
    file_ignore = ['README.md', '.ipynb_checkpoints']
    daily_list = [x.name for x in sorted((repopath / 'dailies').iterdir())
                  if x.name not in file_ignore]
    if 'raw_dailies' in list(x.name for x in (covidpath / 'data').iterdir()):
        make_folders(path, daily_list)
        get_txt_data(path, repopath, daily_list)
    else:
        dailypath = path / 'covid_disinfo_detect' / 'data' / 'raw_dailies'
        dailypath.mkdir()
        make_folders(path, daily_list)
        get_txt_data(path, repopath, daily_list)


def json_check(path, daily_list):
    nojsondates = []
    for day in daily_list:
        daypath = path / day
        files = [x.name for x in daypath.iterdir()
                 if x.name != '.ipynb_checkpoints']
        if f'{day}_clean-dataset.json' in files:
            pass
        else:
            nojsondates.append(day)
    return nojsondates


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


def twarc_gather(path, daily_list):
    for day in daily_list:
        daypath = path / day
        twarc_command = f'''
        twarc hydrate {daypath}/{day}_clean-dataset.txt
        > {daypath}/{day}_clean-dataset.json
        '''
        gzip_command = f'gzip -k {daypath}/{day}_clean-dataset.json'
        print(f'Hydrating tweets for {day}...')
        try:
            subprocess.call(twarc_command, shell=True)
            print('Done gathering data via twarc, compressing JSON...')
            subprocess.call(gzip_command, shell=True)
            print('File compressed! Uploading JSON file to Storage Bucket...')
            upload_blob(
                bucket_name='thepanacealab_covid19twitter',
                source_file_name=f'{daypath}/{day}_clean-dataset.json',
                destination_blob_name=f'dailies/{day}/{day}_clean-dataset.json'
            )
            print('JSON file uploaded to Storage Bucket...')
            print(f'Now removing JSON from {day} folder...')
            filepath = daypath / f'{day}_clean-dataset.json'
            # remove JSON file
            filepath.unlink()
            print(f'JSON removed from {day} folder!')
        except Exception as e:
            print(e)


def main_gather():
    path = setpath()
    covidrawpath = path / 'covid_disinfo_detect' / 'data' / 'raw_dailies'
    daily_list = [x.name for x in sorted((covidrawpath).iterdir())
                  if x.name != 'README.md']
    nojsondates = json_check(covidrawpath, daily_list)
    print(f'Folders w/o JSON: {len(nojsondates)}')
    print(f'\n{nojsondates}')
    totalgather = int(input('How many days would you like to gather?\n'))
    twarc_gather(covidrawpath, nojsondates[-totalgather:])


def main_program():
    main_setup()
    main_gather()


if __name__ == '__main__':
    main_program()

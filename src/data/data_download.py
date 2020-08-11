# Data Ingestion Python Script
import git
from pathlib import Path
import pandas as pd
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
        # create path variables to access data in Panacea repo
        # and path to local storage folder
        storagepath = myrepopath / 'data' / 'raw_dailies' / day
        datapath = panacearepopath / 'dailies' / day
        # get list of contents within local daily storage folder
        files = [x.name for x in storagepath.iterdir()]
        # if daily txt file in daily storage folder print confirmation
        if f'{day}_clean-dataset.txt' in files:
            pass
        # else read in compressed tsv file with Tweet IDs from Panacea repo
        # & store txt file with Tweet IDs in local daily storage folder
        else:
            df = pd.read_csv(
                f'{datapath}/{day}_clean-dataset.tsv.gz',
                sep='\t',
                usecols=['tweet_id'],
                compression='gzip'
            )
            df.to_csv(
                f'{storagepath}/{day}_clean-dataset.txt',
                header=None,
                index=None
            )


def main_setup():
    # set up path to current working directory & path to directory
    # containing Panacea data
    homepath = setpath()
    myrepopath = Path.cwd().parent.parent
    panacearepopath = homepath / 'thepanacealab_covid19'
    if myrepopath.exists():
        pass
    else:
        myrepopath.mkdir()
    # if Panacea lab folder in working directory, print confirmation
    # else clone the repo
    if 'thepanacealab_covid19' in [x.name for x in homepath.iterdir()]:
        print('Panacea Labs COVID-19 GitHub has already been cloned...')
    else:
        clone_panacea_repo(homepath)

    # pull any recent updates from Panacea Lab repo
    pull_result = panacea_pull(panacearepopath)
    print(pull_result)
    # create list of daily folders located in Panacea repo
    file_ignore = ['README.md', '.ipynb_checkpoints']
    daily_list = [
        x.name for x in sorted((panacearepopath / 'dailies').iterdir())
        if x.name not in file_ignore
    ]
    # check to see if data sub-directory exists in my repo
    mydatapath = myrepopath / 'data'
    if mydatapath.exists():
        pass
    else:
        mydatapath.mkdir()

    # if raw_dailies sub-folder exists make folders for raw data, get IDs
    if 'raw_dailies' in list(x.name for x in mydatapath.iterdir()):
        make_raw_folders(myrepopath, daily_list)
        get_txt_data(myrepopath, panacearepopath, daily_list)
    # else make raw_dailies folder then make folders for raw data, get IDs
    else:
        mydailypath = mydatapath / 'raw_dailies'
        mydailypath.mkdir()
        make_raw_folders(myrepopath, daily_list)
        get_txt_data(myrepopath, panacearepopath, daily_list)

    # check if processed_dailies sub-folder exists then create daily folders
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
        src_file_name1 = f'dailies/{day}/{day}_clean-dataset.json'
        src_file_name2 = f'dailies/{day}/panacealab_{day}_clean-dataset.json'
        json1_exist = blob_exists(bucket_name, src_file_name1)
        json2_exist = blob_exists(bucket_name, src_file_name2)
        if json1_exist or json2_exist is True:
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


def twarc_gather(myrawdatapath, daily_list):
    for day in daily_list:
        daypath = myrawdatapath / day
        twarc_command = (
            f'twarc hydrate {daypath}/{day}_clean-dataset.txt > {daypath}/{day}_clean-dataset.json'
        )
        try:
            print(f'Hydrating data for {day}...')
            subprocess.call(twarc_command, shell=True)
            print('Done hydrating data, uploading to bucket...')
            upload_blob(
                bucket_name='thepanacealab_covid19twitter',
                source_file_name=f'{daypath}/{day}_clean-dataset.json',
                destination_blob_name=f'dailies/{day}/{day}_clean-dataset.json'
            )
            print(
                'JSON file uploaded to Storage Bucket,',
                f'now removing JSON from {day} folder...'
            )
            filepath = daypath / f'{day}_clean-dataset.json'
            # remove JSON file
            filepath.unlink()
            print(f'JSON removed from {day} folder!')
        except Exception as e:
            print(e)


def main_gather():
    # set up path to current working directory &
    # path to directory containing Panacea data
    homepath = setpath()
    myrepopath = Path.cwd().parent.parent
    panacearepopath = homepath / 'thepanacealab_covid19'
    myrawdatapath = myrepopath / 'data' / 'raw_dailies'
    # create list of daily folders located in Panacea repo
    file_ignore = ['README.md', '.ipynb_checkpoints']
    daily_list = [
        x.name for x in sorted((panacearepopath / 'dailies').iterdir())
        if x.name not in file_ignore
    ]
    # see what daily data we do not have in storage bucket
    nojson = storage_check(daily_list)
    print(
        f'\nTotal of {len(nojson)} folders do not contain a JSON file:\n{nojson}\n'
    )
    print(
        f'Gathering data for the previous days without JSONs:\n{nojson[::-1]}'
    )
    twarc_gather(myrawdatapath, nojson[::-1])


def main_program():
    main_setup()
    main_gather()


if __name__ == '__main__':
    main_program()

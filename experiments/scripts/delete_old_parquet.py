from google.cloud import storage
import sys
from pathlib import Path
sys.path.insert(0, f'{Path.cwd()}/')
from config import COLS_SELECT, COLS_INTEREST, BUCKET_NAME



def list_parquet_files(bucket_name=BUCKET_NAME):
    """
    Gathers the data of parquet files already stored in GCS
    """
    # init GCS client, get all blobs in bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    
    # list of dates with "cleaned up" raw data
    parquet_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_tweets.parquet')
    ]
    
    return parquet_files


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # print(blob.exists())
    blob.delete()

    print(f'Blob {blob_name} deleted.\n')


def main():
    """
    Main application: deletes irrelevant parquet files from GCS
    """
    parquet_files = list_parquet_files()
    
    for blob in parquet_files:
        blob_to_delete = f'{blob}'
        delete_blob(BUCKET_NAME, blob_to_delete)
        


if __name__ == "__main__":
    main()
    
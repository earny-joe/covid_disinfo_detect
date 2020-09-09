from google.cloud import storage
from settings.config import BUCKET_NAME


def list_raw_json_files(bucket_name=BUCKET_NAME):
    """
    Gathers the data of raw JSON files already stored in GCS
    """
    # init GCS client, get all blobs in bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')

    # list of dates with "cleaned up" raw data
    json_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_clean-dataset.json')
    ]

    return json_files


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print(f'{blob_name}: \t{blob.exists()}\n')


def main():
    """
    Main application: deletes irrelevant raw JSON files from GCS
    """
    json_files = list_raw_json_files()
    N_FILES = 0

    for blob in json_files:
        blob_to_delete = f'{blob}'
        delete_blob(BUCKET_NAME, blob_to_delete)
        N_FILES += 1

    print(f'Num. of files deleted: {N_FILES}')


if __name__ == "__main__":
    main()

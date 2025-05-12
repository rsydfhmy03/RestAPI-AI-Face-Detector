from google.cloud import storage
import uuid

def upload_to_gcs(bucket_name: str, file_bytes: bytes, destination_blob_name: str, content_type: str = "image/jpeg") -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(file_bytes, content_type=content_type)

    return blob.public_url

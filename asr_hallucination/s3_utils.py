"""S3 utilities for working with AWS S3."""

import io
import os

try:
    from botocore.exceptions import ClientError
except ImportError:  # boto3/botocore is optional (only needed for AWS)
    ClientError = Exception

from .config import setup_logging


def parse_s3_uri(uri: str):
    """Parse an S3 URI into bucket and key.

    Args:
        uri: S3 URI of the form s3://bucket/key

    Returns:
        tuple of (bucket, key)
    """
    parts = uri[5:].split("/")
    bucket = parts[0]
    file = os.path.join(*parts[1:]) if len(parts) > 1 else ""
    return bucket, file


def open_read(session, uri: str):
    """Open an S3 object for reading.

    Args:
        session: boto3 session
        uri: S3 URI

    Returns:
        BytesIO buffer with object contents
    """
    bucket, key = parse_s3_uri(uri)
    resource = session.resource('s3')
    obj = resource.Object(bucket, key)
    buf = io.BytesIO()
    obj.download_fileobj(buf)
    buf.seek(0)
    return buf


def delete_uri(session, uri: str):
    """Delete an S3 object.

    Args:
        session: boto3 session
        uri: S3 URI
    """
    bucket, key = parse_s3_uri(uri)
    resource = session.resource('s3')
    resource.Object(bucket, key).delete()


def download_file(session, uri: str, output_file_path=None):
    """Download a file from S3.

    Args:
        session: boto3 session
        uri: S3 URI
        output_file_path: local path to save file (defaults to key filename)

    Returns:
        bool indicating success
    """
    bucket, key = parse_s3_uri(uri)
    client = session.client('s3')
    if output_file_path is None:
        output_file_path = key.split("/")[-1]
    try:
        with open(output_file_path, "wb") as file:
            client.download_fileobj(bucket, key, file)
    except ClientError as e:
        return False
    return True


def upload_file(session, uri: str, file_path: str):
    """Upload a file to S3.

    Args:
        session: boto3 session
        uri: S3 URI destination
        file_path: local file path to upload

    Returns:
        bool indicating success
    """
    logger = setup_logging()
    bucket, key = parse_s3_uri(uri)
    client = session.client('s3')
    try:
        client.upload_file(file_path, bucket, key)
    except ClientError as e:
        logger.error(f"Could not upload the file: {e}")
        return False
    return True

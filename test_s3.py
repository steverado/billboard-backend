import os
import time
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit("Missing python-dotenv. Install with: pip install python-dotenv")

# 1) Load .env
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
USE_PRESIGNED = os.getenv("USE_PRESIGNED_URLS", "false").lower() == "true"

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET]):
    raise SystemExit(
        "Missing required env vars. Check .env for AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_S3_BUCKET."
    )


def _reveal(name: str, val: str):
    # show length + escaped last 12 chars to catch hidden chars like \r, \u200b
    tail = repr(val[-12:] if val else "")
    print(f"{name}: len={len(val) if val else 0}, tail={tail}")


_reveal("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID or "")
_reveal("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY or "")
print("Region:", AWS_REGION, "| Bucket:", AWS_S3_BUCKET)


# 2) Clients
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)
s3 = session.client("s3")
sts = session.client("sts")


def s3_public_url(bucket: str, region: str, key: str) -> str:
    # Works for standard regions; if you're in us-east-1, this format still works
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def main():
    # 3) Verify identity
    ident = sts.get_caller_identity()
    print(f"STS OK. Account: {ident['Account']} | ARN: {ident['Arn']}")

    # 4) Confirm bucket is reachable
    try:
        s3.head_bucket(Bucket=AWS_S3_BUCKET)
        print(f"S3 OK. Bucket exists: {AWS_S3_BUCKET}")
    except ClientError as e:
        raise SystemExit(f"Bucket check failed for '{AWS_S3_BUCKET}': {e}")

    # 5) Create a small test file
    key = f"test_uploads/test_s3_upload_{int(time.time())}.txt"
    body = b"hello from billboard tool setup\n"
    extra_args = {"ContentType": "text/plain"}

    if not USE_PRESIGNED:
        # If your bucket allows public read, we set ACL so you can hit a public URL
        extra_args["ACL"] = "public-read"

    try:
        s3.put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=body, **extra_args)
        print(f"Uploaded: s3://{AWS_S3_BUCKET}/{key}")
    except (ClientError, NoCredentialsError) as e:
        raise SystemExit(f"Upload failed: {e}")

    # 6) URLs
    public_url = s3_public_url(AWS_S3_BUCKET, AWS_REGION, key)
    print(f"Public URL (requires public bucket or object ACL): {public_url}")

    try:
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_S3_BUCKET, "Key": key},
            ExpiresIn=3600,  # 1 hour
        )
        print(f"Presigned URL (works even if bucket is private): {presigned_url}")
    except ClientError as e:
        print(f"Could not create presigned URL: {e}")


if __name__ == "__main__":
    main()

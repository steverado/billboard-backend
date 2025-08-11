# verify_aws.py
import os, sys, json, time
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

load_dotenv()  # loads .env from current directory

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.getenv("AWS_S3_BUCKET", "")


def tail(s, n=6):
    return s[-n:] if s else ""


def die(msg, code=1):
    print(f"❌ {msg}")
    sys.exit(code)


print(
    f"AWS_ACCESS_KEY_ID: len={len(AWS_ACCESS_KEY_ID)}, tail=...{tail(AWS_ACCESS_KEY_ID)}"
)
print(f"AWS_SECRET_ACCESS_KEY: len={len(AWS_SECRET_ACCESS_KEY)}")
print(f"Region: {AWS_DEFAULT_REGION} | Bucket: {BUCKET}")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    die("Missing AWS credentials in .env")

if not BUCKET:
    die("Missing AWS_S3_BUCKET in .env")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)

try:
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    print(f"✅ STS OK. Account: {ident['Account']} | ARN: {ident['Arn']}")
except ClientError as e:
    die(f"STS failed: {e}")
except NoCredentialsError:
    die("No credentials found (check .env)")

s3 = session.client("s3")

# 1) Head bucket (permission + existence)
try:
    s3.head_bucket(Bucket=BUCKET)
    print(f"✅ S3 OK. Bucket exists: {BUCKET}")
except ClientError as e:
    die(f"head_bucket failed: {e}")

# 2) Put + Get + Delete a small test object
key = f"healthchecks/test-{int(time.time())}.txt"
body = b"ok"
try:
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="text/plain")
    print(f"✅ PutObject OK: s3://{BUCKET}/{key}")

    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read()
    if data == body:
        print("✅ GetObject OK: content matches")
    else:
        die("GetObject content mismatch")

    s3.delete_object(Bucket=BUCKET, Key=key)
    print("🧹 Deleted test object")

    print("🎉 All AWS checks passed.")
except ClientError as e:
    die(f"S3 read/write failed: {e}")

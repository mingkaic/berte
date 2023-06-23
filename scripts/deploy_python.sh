#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SOURCE_CODE="$1";
S3_DIR="$2";

BASE="$SCRIPT_DIR/..";
TARGET="$BASE/deploys";

## zip source code
$SCRIPT_DIR/tar_python.sh $SOURCE_CODE $TARGET;

## upload to S3
aws s3 cp "$TARGET/ec2_deployment.tar.gz" "$S3_DIR/ec2_deployment.tar.gz";

# clean up
rm -rf "$TARGET";

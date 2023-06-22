#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SOURCE_CODE="$1";
S3_DIR="$2";

BASE="$SCRIPT_DIR/..";
TARGET="$BASE/deploys";
EC2_DEPLOY_DIR="$BASE/workspace";

# temporary directories
mkdir -p "$TARGET";
mkdir -p "$EC2_DEPLOY_DIR";

# deploying zips

## AWS scripts
cp -r "$BASE/aws_common" "$EC2_DEPLOY_DIR/aws_common";
cp "$BASE/aws/test.py" "$EC2_DEPLOY_DIR";
cp "$BASE/aws/gpumon.py" "$EC2_DEPLOY_DIR";
cp "$BASE/aws_requirements.txt" "$EC2_DEPLOY_DIR";

## training scripts
cp -r "$BASE/common" "$EC2_DEPLOY_DIR/common";
cp "$BASE/requirements.txt" "$EC2_DEPLOY_DIR";
ls -1 $SOURCE_CODE/*.py | awk 'BEGIN { FS = "/" } ; {print $2}' |
	xargs -L1 -I{} cp "$SOURCE_CODE/{}" "$EC2_DEPLOY_DIR/{}";

## zip then upload to S3
pushd $BASE;
tar -cvzf "$TARGET/ec2_deployment.tar.gz" workspace;
aws s3 cp "$TARGET/ec2_deployment.tar.gz" "$S3_DIR/ec2_deployment.tar.gz";
popd;

# clean up
rm -rf "$EC2_DEPLOY_DIR";
rm -rf "$TARGET";

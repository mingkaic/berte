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
cp -r "$BASE/common" "$EC2_DEPLOY_DIR/common";
cp "$BASE/requirements.txt" "$EC2_DEPLOY_DIR";
ls -1 $SOURCE_CODE/*.py | awk 'BEGIN { FS = "/" } ; {print $2}' |
	xargs -L1 -I{} cp "$SOURCE_CODE/{}" "$EC2_DEPLOY_DIR/{}";
tar -cvzf "$TARGET/ec2_deployment.tar.gz" workspace;

pushd $SOURCE_CODE;
tar -cvzf "$TARGET/s3_configs.tar.gz" configs;
tar -cvzf "$TARGET/s3_export.tar.gz" "export";
popd;

aws s3 cp "$TARGET/ec2_deployment.tar.gz" "$S3_DIR/ec2_deployment.tar.gz";
aws s3 cp "$TARGET/s3_configs.tar.gz" "$S3_DIR/s3_configs.tar.gz";
aws s3 cp "$TARGET/s3_export.tar.gz" "$S3_DIR/s3_export.tar.gz";

# clean up
rm -rf "$EC2_DEPLOY_DIR";
rm -rf "$TARGET";

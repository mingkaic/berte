#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SOURCE_CODE="$1";
S3_DIR="$2";

BASE="$SCRIPT_DIR/..";
TARGET="$BASE/deploys";
EC2_DEPLOY_DIR="$BASE/workspace";

# temporary directories
mkdir -p "$TARGET";

# deploying zips

## zip configs and export model
pushd $SOURCE_CODE;
tar -hcvzf "$TARGET/s3_configs.tar.gz" configs;
tar -hcvzf "$TARGET/s3_export.tar.gz" "export";
if [ -d intake ]; then
	tar -hcvzf "$TARGET/s3_intake.tar.gz" intake;
fi
popd;

## upload to S3
aws s3 cp "$TARGET/s3_configs.tar.gz" "$S3_DIR/s3_configs.tar.gz";
aws s3 cp "$TARGET/s3_export.tar.gz" "$S3_DIR/s3_export.tar.gz";
if [ -f "$TARGET/s3_intake.tar.gz" ]; then
	aws s3 cp "$TARGET/s3_intake.tar.gz" "$S3_DIR/s3_intake.tar.gz";
fi

# clean up
rm -rf "$TARGET";

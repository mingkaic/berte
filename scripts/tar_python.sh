#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SOURCE_CODE="$1";
TARGET="$2";

mkdir -p "$TARGET";

BASE="$SCRIPT_DIR/..";
TMP_DIR="$BASE/workspace";

# temporary directories
mkdir -p "$TMP_DIR";

# move AWS scripts
cp -r "$BASE/aws_common" "$TMP_DIR/aws_common";
cp "$BASE/aws_requirements.txt" "$TMP_DIR";
cp "$BASE/aws/gpumon.py" "$TMP_DIR";

# move training scripts
cp -r "$BASE/common" "$TMP_DIR/common";
cp "$BASE/requirements.txt" "$TMP_DIR";
ls -1 $SOURCE_CODE/*.py | awk 'BEGIN { FS = "/" } ; {print $2}' |
	xargs -L1 -I{} cp "$SOURCE_CODE/{}" "$TMP_DIR/{}";

# zip
pushd $BASE;
tar -cvzf "$TARGET/ec2_deployment.tar.gz" workspace;
popd;

# clean up
rm -rf "$TMP_DIR";

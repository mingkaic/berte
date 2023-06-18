#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SOURCE_CODE="$1";

TMP="$SCRIPT_DIR/tmp";
DEPLOY_DIR="$TMP/workspace";

mkdir -p $DEPLOY_DIR;
cp -r "$SCRIPT_DIR/../common" "$DEPLOY_DIR/common";
cp -r "$SOURCE_CODE/configs" "$DEPLOY_DIR/configs";
cp -r "$SOURCE_CODE/export" "$DEPLOY_DIR/export";
cp "$SOURCE_CODE/*.py" "$DEPLOY_DIR";
cp "$SCRIPT_DIR/../requirements.txt" "$DEPLOY_DIR";

tar -cvzf deployment.tar.gz "$DEPLOY_DIR";
rm -rf $TMP
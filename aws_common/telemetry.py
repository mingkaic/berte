#!/usr/bin/env python3
"""
This module includes cloudwatch reporting functions
"""

import boto3

from aws_common.instance import get_instance

#Instance information
_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
IMAGE_ID = _instance_info['image_id']
INSTANCE_TYPE = _instance_info['instance_type']
INSTANCE_AZ = _instance_info['instance_az']
EC2_REGION = _instance_info['ec2_region']

cloudwatch = boto3.client('cloudwatch', region_name=EC2_REGION)
instance_dimensions=[
    {
        'Name': 'InstanceId',
        'Value': INSTANCE_ID
    },
    {
        'Name': 'ImageId',
        'Value': IMAGE_ID
    },
    {
        'Name': 'InstanceType',
        'Value': INSTANCE_TYPE
    },
]

def get_cloudwatch_metric_reporter(namespace, store_reso):
    """ Return metric report function by calling cloudwatch for specified namespace and with
    storage resolution """
    def _report_metric(**kwargs):
        data = [{
            'MetricName': key,
            'Dimensions': instance_dimensions,
            'Unit': 'Percent',
            'StorageResolution': store_reso,
            'Value': value,
        } for key, value in kwargs.items()]
        cloudwatch.put_metric_data(MetricData=data, Namespace=namespace)
    return _report_metric

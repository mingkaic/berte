#!/usr/bin/env python3
"""
This module should run from within an ec2 instance with typical aws accompaniments
"""

from urllib.request import Request, urlopen

from common.builder import Builder

TOKEN_URL = 'http://169.254.169.254/latest/api/token'
BASE_URL = 'http://169.254.169.254/latest/meta-data/'

def _get_token():
    return urlopen(Request(TOKEN_URL,
        headers={ 'X-aws-ec2-metadata-token-ttl-seconds': 21600 },
        method='PUT')).read().decode()

def instance_info_builder():
    """ create Builder for instance info """
    return Builder([
        'image_id',
        'instance_id',
        'instance_type',
        'instance_az',
        'ec2_region',
    ])

def get_instance():
    """ returns the current instance id, image id, type, az, and region """
    headers = { 'X-aws-ec2-metadata-token': _get_token() }
    instance_id = urlopen(Request(BASE_URL + 'instance-id', headers=headers)).read().decode()
    image_id = urlopen(Request(BASE_URL + 'ami-id', headers=headers)).read().decode()
    instance_type = urlopen(Request(BASE_URL + 'instance-type', headers=headers)).read().decode()
    instance_az = urlopen(Request(BASE_URL + 'placement/availability-zone',
        headers=headers)).read().decode()
    ec2_region = instance_az[:-1]
    return instance_info_builder().\
            instance_id(instance_id).\
            image_id(image_id).\
            instance_type(instance_type).\
            instance_az(instance_az).\
            ec2_region(ec2_region).build()

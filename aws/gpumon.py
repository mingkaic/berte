"""
This module is based off https://s3.amazonaws.com/aws-bigdata-blog/artifacts/GPUMonitoring/gpumon.py
"""

import argparse
from time import sleep
from datetime import datetime

import boto3
import pynvml

from aws_common.instance import get_instance

#Instance information
_instance_info = get_instance()
INSTANCE_ID = _instance_info['instance_id']
IMAGE_ID = _instance_info['image_id']
INSTANCE_TYPE = _instance_info['instance_type']
INSTANCE_AZ = _instance_info['instance_az']
EC2_REGION = _instance_info['ec2_region']

TIMESTAMP = datetime.now().strftime('%Y-%m-%dT%H')
TMP_FILE = '/tmp/GPU_TEMP'
TMP_FILE_SAVED = TMP_FILE + TIMESTAMP

# Create CloudWatch client
cloudwatch = boto3.client('cloudwatch', region_name=EC2_REGION)

class GPUMetricsCollector:
    """ GPUMetricsCollector collects and sends metrics to cloudwatch """
    def __init__(self, namespace, store_reso):
        self.push_to_cw = True
        self.namespace = namespace
        self.store_reso = store_reso

    def get_power_draw(self, handle):
        """ return power draw info """
        try:
            pow_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            pow_draw_str = '%.2f' % pow_draw
        except pynvml.NVMLError as err:
            pow_draw_str = pynvml.handleError(err)
            self.push_to_cw = False
        return pow_draw_str

    def get_temp(self, handle):
        """ return temperature info """
        try:
            temp = str(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except pynvml.NVMLError as err:
            temp = pynvml.handleError(err)
            self.push_to_cw = False
        return temp

    def get_utilization(self, handle):
        """ return utilization info """
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = str(util.gpu)
            mem_util = str(util.memory)
        except pynvml.NVMLError as err:
            error = pynvml.handleError(err)
            gpu_util = error
            mem_util = error
            self.push_to_cw = False
        return util, gpu_util, mem_util

    def collect(self, i, handle):
        """ send all get metrics to cloudwatch """
        pow_draw_str = self.get_power_draw(handle)
        temp = self.get_temp(handle)
        util, gpu_util, mem_util = self.get_utilization(handle)

        try:
            gpu_logs = open(TMP_FILE_SAVED, 'a+')
            write_string = str(i) + ',' + gpu_util + ',' +\
                    mem_util + ',' + pow_draw_str + ',' + temp + '\n'
            gpu_logs.write(write_string)
        except:
            print("Error writing to file ", gpu_logs)
        finally:
            gpu_logs.close()
        if self.push_to_cw:
            my_dimensions=[
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
                        {
                            'Name': 'GPUNumber',
                            'Value': str(i)
                        }
                    ]
            cloudwatch.put_metric_data(
                MetricData=[
                    {
                        'MetricName': 'GPU Usage',
                        'Dimensions': my_dimensions,
                        'Unit': 'Percent',
                        'StorageResolution': self.store_reso,
                        'Value': util.gpu
                    },
                    {
                        'MetricName': 'Memory Usage',
                        'Dimensions': my_dimensions,
                        'Unit': 'Percent',
                        'StorageResolution': self.store_reso,
                        'Value': util.memory
                    },
                    {
                        'MetricName': 'Power Usage (Watts)',
                        'Dimensions': my_dimensions,
                        'Unit': 'None',
                        'StorageResolution': self.store_reso,
                        'Value': float(pow_draw_str)
                    },
                    {
                        'MetricName': 'Temperature (C)',
                        'Dimensions': my_dimensions,
                        'Unit': 'None',
                        'StorageResolution': self.store_reso,
                        'Value': int(temp)
                    },
            ],
                Namespace=self.namespace
            )

def main():
    """ main """
    parser = argparse.ArgumentParser(
                    prog='gpumon',
                    description='write gpu metrics to cloudwatch')
    parser.add_argument('--namespace', dest='namespace')
    parser.add_argument('--sleep_interval', dest='sinterval', default=10, type=int)
    parser.add_argument('--store_reso', dest='store_reso', default=60, type=int)

    args = parser.parse_args()

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    try:
        while True:
            collector = GPUMetricsCollector(args.namespace, args.store_reso)
            # Find the metrics for each GPU on instance
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                collector.collect(i, handle)

            sleep(args.sinterval)

    finally:
        pynvml.nvmlShutdown()

if __name__=='__main__':
    main()

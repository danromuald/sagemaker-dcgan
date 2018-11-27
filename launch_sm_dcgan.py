
import sagemaker
import sagemaker_containers
import os
import subprocess
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
# role = sagemaker.get_execution_role() -- This works for EC2.

role = os.environ["SAGEMAKER_EXEC_ROLE"]

instance_type = 'ml.p3.16xlarge' 

inputs = os.environ["DCGAN_INPUTS_DIR"]

dcgan_estimator = PyTorch(entry_point='main-dcgan.py',
                          source_dir='src/',
                          role=role,
                          framework_version='1.0.0.dev',
                          train_instance_count=2,
                          train_instance_type=instance_type,
                          hyperparameters={'epochs': 100,
                                'dist_backend': 'nccl',
                                'display_after': 1000},
                          base_job_name='DCGAN')


dcgan_estimator.fit(inputs=inputs)

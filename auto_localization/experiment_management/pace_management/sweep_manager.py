import argparse
import uuid
import os
import subprocess
import threading
import time
import wandb
import json
from datetime import datetime
import sys
import numpy as np
sys.path.append("../../../..")
from auto_localization.experiment_management.pace_management.plot_generator import PlotGenerator
from auto_localization.experiment_management.hyperparameter_experiment import HyperparameterExperiment
from auto_localization.dataset_management.data_manager import DataManager
from auto_localization.oracles.metadata_oracle import MetadataOracle
from auto_localization.oracles.indexed_metadata_oracle import IndexedMetadataOracle
from auto_localization.oracles.indexed_class_oracle import IndexedClassOracle
from auto_localization.oracles.oracle import EnsembleOracle
from auto_localization.experiment_management import util as experiment_management_util

LOCAL_SCRIPT_TEMPLATE = """
python $LATENT_PATH/auto_localization/experiment_management/pace_management/single_experiment_runner.py {config_path}
"""

SCRIPT_TEMPLATE = """
#PBS -N config_sweep
#PBS -A GT-crozell3-CODA20
#PBS -l nodes=1:ppn=6:gpus=1:RTX6000
#PBS -l pmem=16gb
#PBS -l walltime={wall_time}
#PBS -q {queue_name}
#PBS -j oe
#PBS -o {log_file} 
#PBS -m abe
#PBS -M ahelbling6@gatech.edu
nvidia-smi
cd $PBS_O_WORKDIR
module load anaconda3/2019.10
module load cuda/10.1
source activate latent

CUDA_LAUNCH_BLOCKING=1 python $LATENT_PATH/auto_localization/experiment_management/pace_management/single_experiment_runner.py {config_path}
"""

LOCAL_PLOT_SCRIPT_TEMPLATE = """
python $LATENT_PATH/auto_localization/experiment_management/pace_management/plot_generator.py {group_name} '{experiment_config}'
"""

PLOT_SCRIPT_TEMPLATE = """
#PBS -N sweep_plotting
#PBS -A GT-crozell3
#PBS -l nodes=1:ppn=6:gpus=1:RTX6000
#PBS -l pmem=16gb
#PBS -l walltime={wall_time}
#PBS -q {queue_name}
#PBS -j oe
#PBS -o {log_file} 
#PBS -m abe
#PBS -M ahelbling6@gatech.edu

cd $PBS_O_WORKDIR
module load anaconda3/2019.10
source activate latent
python $LATENT_PATH/auto_localization/experiment_management/pace_management/plot_generator.py {group_name} '{experiment_config}'
"""

"""
    Code to encapsulate a PBS Job
"""
class PBSJob():
        
    def __init__(self, local=False):
        self.file_path = None
        self.local = local
 
    """
        Runs the necessary bash command to queue the
        given script
    """
    def queue(self):
        print("Enqueuening job")
        print(self.file_path)
        if self.local:
            command = f'sh {self.file_path}'
        else:
            command = f'qsub {self.file_path}'
        # run the command
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    """
        Generate a plotting script
    """
    def generate_plotting_script(self, config_directory, group_name, experiment_config, wall_time, queue_name):
        if self.local:
            script_template = LOCAL_PLOT_SCRIPT_TEMPLATE
        else:
            script_template = PLOT_SCRIPT_TEMPLATE
        script_file_path = os.path.join(config_directory, "plotting_script")
        plotting_log = os.path.join(config_directory, "plotting_log")
        config_json = json.dumps(experiment_config)
        script_contents = script_template.format(log_file=plotting_log, wall_time=wall_time, group_name=group_name, experiment_config=config_json, queue_name=queue_name)
        with open(script_file_path, "w") as f:
            f.write(script_contents)
        self.file_path = script_file_path

    """
        Generates a bash script for running a single experiment 
        from the given experiment config file
    """
    def generate_experiment_script(self, config_file, log_path, wall_time, queue_name):
        if self.local:
            script_template = LOCAL_SCRIPT_TEMPLATE
        else:
            script_template = SCRIPT_TEMPLATE
        script_contents = script_template.format(config_path=config_file, log_file=log_path, wall_time=wall_time, queue_name=queue_name)
        script_file_path = config_file+"_script"
        with open(script_file_path, "w") as f:
            f.write(script_contents)
        self.file_path = script_file_path
    
class WaiterProcess():

    def __init__(self, model_configs, num_attempts, group_name, num_trials):
        self.group_name = group_name
        self.model_configs = model_configs
        self.num_trials = num_trials
        self.num_jobs = len(model_configs) * self.num_trials
        self.failure_limit = 5*self.num_jobs
        self.max_jobs = 30 # keep queued or runnning jobs under this number
        self.failure_name_list = []
        self._setup_config_array()
    
    def get_runs(self):
        api = wandb.Api()
        runs = api.runs(path="helblazer811/latent-space-localization", filters={"group": self.group_name})
        return runs
 
    def get_config_index(self, config):
        for i, index_config in enumerate(self.config_array):
            if index_config == config:
                return i

    def _setup_config_array(self):
        self.config_array = []
        for config in self.model_configs:
            if not config in self.config_array:
                self.config_array.append(config)
        print(self.config_array)

    def is_finished(self, finished_index_list):
        index_total = {}    
        for i in range(len(self.config_array)):
            index_total[i] = 0
        print(finished_index_list)
        print(index_total)
        for index in finished_index_list: 
            if index is None:
                continue
            if index in finished_index_list:
                index_total[index] += 1
        print(index_total)
        for i in range(len(self.config_array)):
            if index_total[i] < self.num_trials:
                return False
        print("finished") 
        return True

    """
        Checks the status of the running jobs in wandb and
        the log files to see if one failed or if they are all finished
    """
    def check_status(self):
        # init necessary values
        num_finished = 0
        failed_job_configs = []
        runs = self.get_runs()
        # running/finished map
        running_finished_map = {}
        finished_index_list = []
        # go through each run
        for run in runs: 
            state = run.state
            name = run.name
            model_config = run.config["model_config"]
            index_of_config = self.get_config_index(model_config)
            if not index_of_config in running_finished_map:
                running_finished_map[index_of_config] = 0
            # check the state of the run
            if state == "failed":
                # is it a new failure 
                if name in self.failure_name_list:
                    continue
                self.failure_name_list.append(name) 
                num_failed = len(self.failure_name_list)
                exceed_failure_limit = num_failed > self.failure_limit
                if exceed_failure_limit:
                    break
                print("new failure")
                # handle new failure scenario
                failed_job_configs.append(model_config) 
            elif state == "finished":
                running_finished_map[index_of_config] += 1
                finished_index_list.append(index_of_config)
            elif state == "running":
                running_finished_map[index_of_config] += 1

        # see if too many jobs are already running
        rerun_configs = []
        for failed_config in failed_job_configs:
            index_of_config = self.get_config_index(failed_config)    
            if running_finished_map[index_of_config] < self.num_trials:
                running_finished_map[index_of_config] += 1
                # check if failure number has been exceeded
                rerun_configs.append(failed_config)
        # check if number of finished are equal to the total number of jobs
        finished_jobs = self.is_finished(finished_index_list)
    
        return finished_jobs, rerun_configs  

"""
    Sweep manager object that handles controlling a given hyperparameter sweep
"""
class SweepManager():
    
    def __init__(self, experiment_config, wall_time="8:00:00", queue_name="embers"):
        self.experiment_config = experiment_config 
        self.num_trials = self.experiment_config["trials"]
        self.group_name = self.experiment_config["group"]
        self.queue_name = queue_name
        if "local" in self.experiment_config:
            self.local = self.experiment_config["local"]
        else:
            self.local = False
        self.wall_time = wall_time 
        self.config_directory = "logs/"+self.group_name
        self.log_path = os.path.join(self.config_directory, "log_file.out")
        self.num_attempts = 5
        self.plot_job = None
        self.experiment_jobs = []
        self._setup()
    
    def _make_experiment_jobs(self, config_files):
        jobs = []
        for config_path in config_files: 
            job = PBSJob(local=self.local)
            job.generate_experiment_script(config_path, self.log_path, self.wall_time, self.queue_name)
            jobs.append(job)

        return jobs

    def _make_plot_job(self):
        job = PBSJob(local=self.local)
        job.generate_plotting_script(self.config_directory, self.group_name, self.experiment_config, self.wall_time, self.queue_name)
        return job

    """
        Runs setup code
    """
    def _setup(self):
        print("Running setup")
        # saves sweep config files
        self.model_configs, self.pivot_keys = experiment_management_util.generate_experiment_split(self.experiment_config) 
        print(self.pivot_keys)
        # save the configs
        config_file_paths = experiment_management_util.save_configs(self.model_configs, self.experiment_config, self.config_directory)
        # make a dictionary storing the number of try attempts   
        self.experiment_jobs = self._make_experiment_jobs(config_file_paths)
        self.plot_job = self._make_plot_job()
    
    def run_failed_jobs(self, failed_job_configs):
        print("Running failed jobs")
        experiment_config_copy = self.experiment_config.copy()
        experiment_config_copy["trials"] = 1
        config_file_paths = experiment_management_util.save_configs(failed_job_configs, experiment_config_copy, self.config_directory)
        new_jobs = self._make_experiment_jobs(config_files = config_file_paths)
        print("Num new jobs")
        print(len(new_jobs))
        # queue the jobs
        for job in new_jobs:
            job.queue()

    """
        Starts a loop process that manages the jobs that are run. 
        It also runs a plot job when everything is finished. 
    """
    def _run_wait_process(self):
        num_trials = self.experiment_config["trials"]
        waiter_process = WaiterProcess(self.model_configs, self.num_attempts, self.group_name, num_trials)
        finished = False 
        while not finished:
            time.sleep(10) # sleep for 1 second
            finished, failed_job_configs = waiter_process.check_status() 
            # run failed jobs
            if len(failed_job_configs) > 0:
                # check job number limit 
                self.run_failed_jobs(failed_job_configs)
        print("Finished waiting")
        # run the plot job when finished
        if finished:
            self.plot_job.queue()
            
    """
        Run the sweep
    """
    def run_sweep(self):
        print("Running sweep")
        for experiment_job in self.experiment_jobs:
            experiment_job.queue() 
        # setup meta plotting background process
        self._run_wait_process()


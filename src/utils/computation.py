"""
For ETHZ users:
To use this script, you need to change for each computing node,
the project folder on each computing node and data folder.
"""

import subprocess
from abc import abstractmethod
import itertools


class Node(object):
    """
    Base class for describing computing nodes of experiments.
    For every new computing node, one should make a realization of this class.
    * :attr: 'address': str the address of the computing node
    * :attr: 'data_root': str the home folder of all data on the computing node
    """

    def __init__(self):
        self.address = None
        self.data_root = None
        self.project_root = None

    def get_address(self):
        return self.address

    def get_data_root(self):
        return self.data_root

    def get_project_root(self):
        return self.project_root

    @abstractmethod
    def run_experiment(self, experiment_arguments, gpus):
        """ Do the computation.
        :param experiment_arguments: string of arguments
        :return: [False/True] depending if job submission was successful
        """


class NoNode(Node):
    def run_experiment(self, experiment_arguments, gpus):
        command = 'python train.py' + experiment_arguments
        subprocess.Popen([command],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)


class Euler(Node):
    """ For executing computations on ETHZ Euler. """

    def __init__(self):
        super(Euler, self).__init__()
        self.address = "euler.ethz.ch"
        self.data_scratch = "$TMPDIR/data/"
        self.client = None
        self.project_path = "projects/multi-dose-fselect"

    def node_run_experiment(self,
                            experiment_config,
                            gpus,
                            email,
                            name,
                            mem=None,
                            gpu_model=None,
                            cpu_nodes=1,
                            gpu_q=None,
                            slurm=False,
                            cv_fold=None,
                            full_pipeline=False):

        """ this is a non-blocking run """
        print("Sending a job to Euler.")
        if email is True:
            N = '-N ' if slurm is False else '--mail-type=END'
        else:
            N = ''
        if name != None:
            J = '-J ' + name + ' '
            if slurm is True:
                O = '-o ' + 'outputs/' + name + ' '
            else:
                O = '-oo ' + 'outputs/' + name + ' '
        else:
            J = ''
            O = ''

        # augment experiment arguments

        # Setup specification gpu_model
        gpu_info = ''
        if gpu_q is not None:
            if slurm:
                gpu_info = gpu_info + ' --time=' + str(gpu_q) + ':00:00'
            else:
                gpu_info = gpu_info + ' -W ' + str(gpu_q) + ':00'
        if slurm:
            if mem is not None:
                gpu_info = gpu_info + ' --mem-per-cpu=' + str(mem)
            else:
                gpu_info = gpu_info + ' --mem-per-cpu=10000'
                # gpu_info = gpu_info + ' -R "rusage[mem=11000,ngpus_excl_p=' + str(gpus) + ']" '
            if gpu_model is not None:
                gpu_info = gpu_info + ' --gpus=' + gpu_model + ':' + str(gpus)
            else:
                gpu_info = gpu_info + ' --gpus=' + str(gpus)

        else:
            if mem is not None:
                gpu_info = gpu_info + ' -R "rusage[mem=11000,ngpus_excl_p=' + str(gpus) + ']" '
            else:
                gpu_info = gpu_info + ' -R "rusage[ngpus_excl_p=' + str(gpus) + ']"'
            if gpu_model is not None:
                gpu_info = gpu_info + ' -R "select[gpu_model0==' + gpu_model + ']"'

        if slurm:
            sub = 'sbatch '
        else:
            sub = 'bsub '
        if experiment_config is not None:
            experiment_arguments = 'experiment=' + experiment_config
        else:
            experiment_arguments = ""

        command_req = 'source .bash_profile; cd ' + self.project_path + '; ' \
                'module load gcc/8.2.0; ' \
                'module load python_gpu/3.8.5; ' \
                'module load cuda/11.3.1; ' \
                'module load eth_proxy; ' \
                '' + sub + N + J + O + '-n ' + str(int(cpu_nodes)) + gpu_info + ' '

        if slurm:
            command_req = command_req + '--wrap '

        if cv_fold:

            script_command =  "'"
            for i in range(cv_fold):

                name_cv = name + '_cv' + str(i)

                # Train script
                experiment_arguments_new = experiment_arguments + " name=" + name_cv
                experiment_arguments_new = experiment_arguments_new + " datamodule.cv_fold=" + str(i)
                script_command = script_command + "python " + "train.py " + experiment_arguments_new + "; "

                # Feature Selection script
                experiment_arguments_new = 'experiment='+experiment_config +  " name=" + name_cv
                # experiment_arguments_new = experiment_arguments +  " name=" + name_cv
                script_command = script_command + "python " + "select_features.py " + experiment_arguments_new + "; "


                # # Classifier script

                # experiment_arguments_new = 'experiment=classifier' +  " name=" + name_cv
                # script_command = script_command + "python " + "train_classifier.py "  + experiment_arguments_new + "; "
                # Multiclassifier
                script_command = script_command + "python multiple_classifier.py -name_cv " + name_cv


            command = command_req + script_command + "' "
        else:
            command = command_req + "'python " + 'train.py' + ' ' + experiment_arguments + "' "

        print(command)
        print("Sleep for 2 seconds..")
        import time
        time.sleep(2)

        # submit the job to Leonhard
        subprocess.Popen(["ssh", "%s" % "jcarvalho@euler.ethz.ch", command],
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        print("Job sent!")

        return True


# def run_experiment(nodes, exp_name, models, repetitions, parameters):
def run_experiment(nodes, experiment, gpus, email, name, mem, cpu_nodes, gpu_model, gpu_q, slurm=False,cv_fold=None):
    # find a computing node and run the experiment on it
    for node in nodes:
        print("Trying node: ",
              node.address)  # experiment_config, gpus, email, name, run_script,mem=None,gpu_model=None,gpu_q=None)
        success = node.node_run_experiment(experiment_config=experiment,
                                           gpus=gpus,
                                           email=email,
                                           name=name,
                                           mem=mem,
                                           cpu_nodes=cpu_nodes,
                                           gpu_model=gpu_model,
                                           gpu_q=gpu_q,
                                           slurm=slurm,
                                           cv_fold=cv_fold)
        if success:
            print("Connection established!")
            break
        else:
            print("This node is busy.")

"""
Loosely based on : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

search_space = {
    'num_hidden': {'_type': 'choice', '_value': [16, 32, 64, 128]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 0.1, 1]},
}

from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 2
experiment.run(8080)


input('Press enter to quit')
experiment.stop()

# After the experiment is stopped, you can run :meth:`nni.experiment.Experiment.view` to restart web portal.

"""
Loosely based on : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

ACTIVATIONS = ['ELU','Hardshrink','Hardsigmoid','Hardtanh','Hardswish',
               'LeakyReLu','LogSigmoid','ReLU','ReLU6','RReLU','Sigmoid',
               'SiLU','Mish','Softplus','Softshrink','Softsign','Tanh','Tanhshrink',
               'Threshold']

ACTIVATIONS_0 = ['SoftSign','LogSigmoid','Tanhshrink','Hardtanh','Hardsigmoid','ReLU6','SoftPlus','SiLU']

search_space = {
    'num_layers':{'_type': 'choice','_value':[1,2,3]},
    'num_1': {'_type': 'choice', '_value': [1, 2, 4, 8]},
    'num_2': {'_type': 'choice', '_value': [1, 2, 4, 8, 16, 32, 64, 128]},
    'num_3': {'_type': 'choice', '_value': [1, 2, 4, 8]},
    'act_0': {'_type': 'choice', '_value': ACTIVATIONS},
    'act_1': {'_type':'choice','_value':ACTIVATIONS},
    'act_2': {'_type': 'choice', '_value': ACTIVATIONS},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 0.1, 1]},
}

from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 10000
experiment.config.trial_concurrency = 2
experiment.run(8080)


input('Press enter to quit')
experiment.stop()

# After the experiment is stopped, you can run :meth:`nni.experiment.Experiment.view` to restart web portal.

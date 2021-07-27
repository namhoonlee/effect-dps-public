import os
import json


def setup_study(args):
    filename='./study.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            study = json.load(f)
    else:
        study = {
            'sparsity': args.target_sparsity,
            'batch_size': args.batch_size,
            'goal_error': args.goal_error,
            'nruns': args.nruns,
            'check_interval': args.check_interval_arrival,
            'mparams_kind': args.mparams_kind,
            'dataset': args.datasource,
            'model': args.arch,
            'optimizer': args.optimizer,
            'parameter_configs': {
                'learning_rate': {
                    'max_value': 1.0,
                    'min_value': 0.0001,
                    'scale': 'LOG_SCALE',
                    'type': 'DOUBLE',
                },
                'momentum': {
                    'max_value': 0.9999,
                    'min_value': 0.9,
                    'scale': 'REVERSE_LOG_SCALE',
                    'type': 'DOUBLE',
                },
                'decay_steps':{
                    'max_value': 60000,
                    'min_value': 30000,
                    'scale': '',
                    'type': 'INTEGER',
                },
                'end_learning_rate_factor':{
                    'max_value': 0.1,
                    'min_value': 0.0001,
                    'scale': 'LOG_SCALE',
                    'type': 'DOUBLE',
                },
            },
        }
        with open(filename, 'w') as f:
            json.dump(study, f, indent=2)
    return study

def save_results(args, steps_to_result, status):
    metadata = {
        'parameters': {key: getattr(args, key) for key in args.mparams_kind},
        'steps-to-result': steps_to_result,
        'status': status,
    }
    if not os.path.isdir(args.path['batch-science']):
        os.makedirs(args.path['batch-science'])
    filename = os.path.join(args.path['batch-science'], 'metadata.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)

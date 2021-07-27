import os
import functools
import json

from sobol_lib import *


def load(args, study):
    filename='./metaparameters.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            mparams = json.load(f)
    else:
        mparams = _sample(args.mparams_kind, args.nruns, study)
        with open(filename, 'w') as f:
            json.dump(mparams, f, indent=2)
    assert set(mparams.keys()) == set(args.mparams_kind)
    return mparams

def _sample(mparams_kind, nsample, study):
    sequences = i4_sobol_generate(len(mparams_kind), nsample, skip=2)
    mparams = {}
    for i, kind in enumerate(mparams_kind):
        sequence = sequences[i]
        configs = study['parameter_configs'][kind]
        if configs['scale'] == 'LOG_SCALE':
            sequence_new = [
                _unit_to_log10scale(val, configs['max_value'], configs['min_value'])
                for val in sequence]
        elif configs['scale'] == 'REVERSE_LOG_SCALE':
            sequence_new = [
                _unit_to_reverselog10scale(val, configs['max_value'], configs['min_value'])
                for val in sequence]
        elif configs['scale'] == '':
            print('debug')
            sequence_new = [
                _unit_to_scaleshift(val, configs['max_value'], configs['min_value'],
                    configs['type'])
                for val in sequence]
        else:
            raise NotImplementedError
        mparams[kind] = sequence_new
    return mparams

def _unit_to_log10scale(val, max_value, min_value):
    scalar = math.log10(max_value) - math.log10(min_value)
    shift = math.log10(min_value)
    return 10 ** (val * scalar + shift)

def _unit_to_reverselog10scale(val, max_value, min_value):
    scalar = math.log10(1 - min_value) - math.log10(1 - max_value)
    shift = math.log10(1 - max_value)
    return 1 - (10 ** (val * scalar + shift))

def _unit_to_scaleshift(val, max_value, min_value, dtype):
    scalar = max_value - min_value
    shift = min_value
    if dtype == 'INTEGER':
        return int(val * scalar + shift)
    else:
        return val * scalar + shift

def update(args, mparams, run):
    print('-- Update metaparameters for the current run ({})'.format(run))
    dargs = vars(args)
    for key, val in mparams.items():
        if key in args:
            dargs[key] = mparams[key][run]
            print('  {}: {}'.format(key, mparams[key][run]))
        else:
            assert False, 'Trigger an assertion error for now (tentative).'
    return args

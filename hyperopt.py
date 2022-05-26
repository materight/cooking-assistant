"""Script to run hyperparameters optimization of a Rasa model."""
import os
import yaml
import argparse
import subprocess
from datetime import datetime

from sklearn.model_selection import ParameterSampler

parser = argparse.ArgumentParser(description="Run hyperparameters optimization of a Rasa model.")
parser.add_argument('--n-iter', '-n', type=int, default=10, help="Total number of iterations to run (default: %(default)s).")

def set_hyperparams(config: dict, params: dict) -> dict:
    """Set the given hyperparams in the config dictionary."""
    if isinstance(config, dict):
        return { k: set_hyperparams(v, params) for k, v in config.items() } # Recursively set hyperparams in dict
    elif isinstance(config, list):
        return [ set_hyperparams(v, params) for v in config ] # Recursively set hyperparams in list
    elif isinstance(config, str) and config.startswith('$'):
        return params[config.lstrip('$')] # Set hyperparameter value
    return config

def listdir(path: str, exclude: list = []):
    """List all files in a directory."""
    for d in os.listdir(path):
        if os.path.isdir(d) and d not in exclude:
            yield d, os.path.join(path, d)

if __name__ == "__main__":
    args = parser.parse_args()
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    work_dir = os.path.join('hyperopts', exp_name)
    os.makedirs(work_dir)
    print(f'Experiment name: {exp_name}')
    
    # Load hyperopt config
    with open('config.hyperopt.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        hyperparams = config['hyperparams']
        del config['hyperparams']
    
    # Generate the config files to compare
    configs = []
    sampler = ParameterSampler(hyperparams, n_iter=args.n_iter, random_state=0)
    os.makedirs(os.path.join(work_dir, 'configs'))
    print(f'Generating {len(sampler)} pipeline configs...')
    for i, params in enumerate(sampler):
        with open(os.path.join(work_dir, 'configs', f'{i+1}.yml'), 'w') as f:
            yaml.dump(set_hyperparams(config, params), f)
            configs.append(f.name)
    
    # Train and test NLU models with cross-validation
    print('Training and testing NLU models...')
    subprocess.run(['rasa', 'test', 'nlu',
        '--config', *configs,
        '--cross-validation',
        '--runs', '3',
        '--out', f'{work_dir}/nlu',
        '--model', f'{work_dir}/nlu/models'
    ], check=True).returncode
    
    # Train and test dialogue models
    print('Training and testing dialogue models...')
    subprocess.run(['rasa', 'train', 'core',
        '--config', *configs,
        '--cross-validation',
        '--runs', '3',
        '--out', f'{work_dir}/core/models'
    ], check=True).returncode
    for split, stories_dir in dict(train='data', test='tests').items():  # The previous models have been trained excluding a certain amount of training data, so we can evaluate also over the train set
        subprocess.call(['rasa test core',
            '--model', f'{work_dir}/core/models',
            '--stories', stories_dir,
            '--runs', '3',
            '--evaluate-model-directory'
            '--out', f'{work_dir}/core/{split}'
        ], check=True).returncode
    
    # Read results from the output files
#    for run_name, run_path in listdir(os.path.join(work_dir, 'nlu')):
#        for fold_name, fold_path in listdir(run_path):
#            for model_name, model_path in listdir(fold_path, exclude='train'):
                

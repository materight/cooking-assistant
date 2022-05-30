"""Script to run hyperparameters optimization of a Rasa model."""
import os
import glob
import yaml
import json
import argparse
import subprocess
from datetime import datetime
from collections import defaultdict

from sklearn.model_selection import ParameterSampler

parser = argparse.ArgumentParser(description="Run hyperparameters optimization of a Rasa model.")
parser.add_argument('--n-iter', '-n', type=int, default=10, help="Total number of iterations to run (default: %(default)s).")

PROJECT_ROOT = os.path.dirname(__file__)

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
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if os.path.isdir(dir_path) and dir_name not in exclude:
            yield dir_name, dir_path


def run_hyperopts(exp_name: str, n_iter: int):
    """Run hyperparameters search."""
    work_dir = os.path.join(PROJECT_ROOT, 'hyperopts', exp_name)
    os.makedirs(work_dir)
    print(f'Experiment name: {exp_name}')
    
    # Load hyperopt config
    with open('config.hyperopt.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        hyperparams = config['hyperparams']
        del config['hyperparams']
    
    # Generate the config files to compare
    configs = []
    sampler = ParameterSampler(hyperparams, n_iter=n_iter, random_state=0)
    os.makedirs(os.path.join(work_dir, 'configs'))
    print(f'Generating {len(sampler)} pipeline configs...')
    for i, params in enumerate(sampler):
        with open(os.path.join(work_dir, 'configs', f'{i+1}.yml'), 'w') as f:
            yaml.dump(dict(**set_hyperparams(config, params), hyperparams=params), f)
            configs.append(f.name)
    
    # Train and test NLU models with cross-validation
    print('Training and testing NLU models...')
    subprocess.run(['rasa', 'test', 'nlu',
        '--config', *configs,
        '--cross-validation',
        '--no-plot',
        '--runs', '3',
        '--percentages', '0', '50', '75',
        '--out', f'{work_dir}/nlu',
        '--model', f'{work_dir}/nlu/models'
    ], check=True).returncode
    
    # Train and test dialogue models
    print('Training and testing dialogue models...')
    subprocess.run(['rasa', 'train', 'core',
        '--config', *configs,
        '--cross-validation',
        '--runs', '3',
        '--percentages', '0', '50', '75',
        '--out', f'{work_dir}/core/models'
    ], check=True).returncode
    for split, stories_dir in dict(train='data', test='tests').items():  # The previous models have been trained excluding a certain amount of training data, so we can evaluate also over the train set
        subprocess.call(['rasa test core',
            '--model', f'{work_dir}/core/models',
            '--stories', stories_dir,
            '--no-plot',
            '--runs', '3',
            '--evaluate-model-directory'
            '--out', f'{work_dir}/core/{split}'
        ], check=True).returncode
    
    # Delete generated models and plots to save space
    for model_filepath in glob.glob(os.path.join(work_dir, '**/*.tar.gz'), recursive=True):
        os.remove(model_filepath)
    for plot_filepath in glob.glob(os.path.join(work_dir, '**/*.png'), recursive=True):
        os.remove(plot_filepath)
    

def process_results(exp_name):
    """Process the results of hyperparams search."""
    # Read results from the output files
    work_dir = os.path.join(PROJECT_ROOT, 'hyperopts', exp_name)
    # Parse NLU results
    runs_paths = list(listdir(os.path.join(work_dir, 'nlu')))
    runs_count = len(runs_paths)
    nlu_results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for run_name, run_path in runs_paths:
        for fold_name, fold_path in listdir(run_path):
            exclusion_fraction = fold_name.replace('_exclusion', '')
            for report_name, report_path in listdir(fold_path, exclude='train'):
                config_name = report_name.replace('_report', '')
                # Get report files of each component
                for component_report_path in glob.glob(os.path.join(report_path, '*_report.json')):
                    component_name = os.path.basename(component_report_path).replace('_report.json', '')
                    with open(component_report_path, 'r') as f:
                        component_report = json.load(f)
                    f1_score = component_report['weighted avg']['f1-score']
                    nlu_results[config_name][component_name][exclusion_fraction] += f1_score / runs_count # Average over three runs




if __name__ == "__main__":
    args = parser.parse_args()
    exp_name = '20220528-115649' # datetime.now().strftime('%Y%m%d-%H%M%S')
    #run_hyperopts(exp_name, args.n_iter)
    process_results(exp_name)
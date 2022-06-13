"""Script to run hyperparameters optimization of a Rasa model."""
import os
import tempfile
import glob
import yaml
import json
import argparse
import subprocess
from datetime import datetime
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import ParameterSampler

parser = argparse.ArgumentParser(description="Run hyperparameters optimization of a Rasa model.")
parser.add_argument('--n-iter', '-n', type=int, default=100, help="Total number of iterations to run (default: %(default)s).")
parser.add_argument('--n-runs', '-r', type=int, default=3, help="Total number of experiments per run (default: %(default)s).")
parser.add_argument('--percentages', '-p', type=int, nargs="+", default=[0, 50, 75], help="Fractions of training data to held-out during training (default: %(default)s).")
parser.add_argument('--component', '-c', type=str, default=None, choices=['all', 'nlu', 'core'], help="Which component to train (default: %(default)s, choices: %(choices)s).")


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


def run_hyperopts(exp_name: str, n_iter: int, n_runs: int, percentages: list):
    """Run hyperparameters search."""
    work_dir = os.path.join(PROJECT_ROOT, 'hyperopts', exp_name)
    os.makedirs(work_dir)
    print(f'Experiment name: {exp_name}')

    # Load hyperopt config
    with open(os.path.join(PROJECT_ROOT, 'config.hyperopt.yml'), 'r') as f:
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
    if args.component in ['all', 'nlu']:
        print('\nTraining and testing NLU models...')
        subprocess.run(['rasa', 'test', 'nlu',
            '--domain', os.path.join(PROJECT_ROOT, 'domain.yml'),
            '--nlu', os.path.join(PROJECT_ROOT, 'data'),
            '--config', *configs,
            '--cross-validation',
            '--no-plot',
            '--runs', str(n_runs),
            '--percentages', *map(str, percentages),
            '--out',  os.path.join(work_dir, 'nlu'),
            '--model',  os.path.join(work_dir, 'nlu', 'models')
        ], check=True).returncode

    if args.component in ['all', 'core']:
        # Merge stories and rules in a single file to support rasa training with multiple stories
        with tempfile.NamedTemporaryFile(suffix='.yml', mode='w', delete=False) as tmp_file:
            tmp_stories_path = tmp_file.name
            with open(os.path.join(PROJECT_ROOT, 'data', 'stories.yml'), 'r') as stories_file, \
                open(os.path.join(PROJECT_ROOT, 'data', 'rules.yml'), 'r') as rules_file:
                tmp_file.write(stories_file.read())
                rules_file.readline() # Skip first line with "version"
                tmp_file.write(rules_file.read())

        # Train dialogue models
        print('\nTraining dialogue models...')
        subprocess.run(['rasa', 'train', 'core',
            '--domain', os.path.join(PROJECT_ROOT, 'domain.yml'),
            '--stories', tmp_stories_path,
            '--config', *configs,
            '--runs', str(n_runs),
            '--percentages', '0',
            '--out', os.path.join(work_dir, 'core', 'models')
        ], check=True).returncode

        # Test dialogue models
        print('\nTesting dialogue models...')
        for run_name, run_path in listdir(os.path.join(work_dir, 'core', 'models')):
            for model_path in glob.iglob(os.path.join(run_path, '*.tar.gz')):
                model_name = os.path.basename(model_path).replace('.tar.gz', '')
                for split, stories_dir in dict(train='data', test='tests').items():
                    subprocess.run(['rasa', 'test', 'core',
                        '--model', model_path,
                        '--stories', os.path.join(PROJECT_ROOT, stories_dir, 'stories.yml'),
                        '--out', os.path.join(work_dir, 'core', run_name, model_name, split)
                    ], check=True).returncode
 

def process_results(exp_name):
    """Process the results of hyperparams search."""
    # Read results from the output files
    work_dir = os.path.join(PROJECT_ROOT, 'hyperopts', exp_name)
    # Load configs hyperparameters
    configs = {}
    for config_file in glob.iglob(os.path.join(work_dir, 'configs', '*.yml')):
        config_name = int(os.path.basename(config_file).replace('.yml', ''))
        with open(config_file, 'r') as f:
            configs[config_name] = yaml.load(f, Loader=yaml.FullLoader)['hyperparams']
    # Parse NLU results
    if os.path.exists(os.path.join(work_dir, 'nlu')):
        runs_paths = list(listdir(os.path.join(work_dir, 'nlu')))
        runs_count = len(runs_paths)
        nlu_results = defaultdict(lambda: defaultdict(int))
        for run_name, run_path in runs_paths:
            for fold_name, fold_path in listdir(run_path):
                exclusion_fraction = fold_name.replace('_exclusion', '')
                for report_name, report_path in listdir(fold_path, exclude='train'):
                    config_name = int(report_name.replace('_report', ''))
                    nlu_results[config_name].update({ ('', k): v for k, v in configs[config_name].items() })
                    # Get report files of each component
                    for component_report_path in glob.glob(os.path.join(report_path, '*_report.json')):
                        component_name = os.path.basename(component_report_path).replace('_report.json', '')
                        with open(component_report_path, 'r') as f:
                            component_report = json.load(f)
                        f1_score = component_report['weighted avg']['f1-score'] * 100
                        nlu_results[config_name][(component_name, exclusion_fraction)] += f1_score / runs_count # Average over the runs
        nlu_results = pd.DataFrame.from_dict(nlu_results, orient='index').sort_index(axis=1, ascending=True).sort_values((component_name, '0%'), ascending=False)
        nlu_results.to_csv(os.path.join(work_dir, 'nlu_report.csv'), sep='\t', float_format='%.2f')
    # Parse core results
    if os.path.exists(os.path.join(work_dir, 'core')):
        runs_paths = list(listdir(os.path.join(work_dir, 'core'), exclude='models'))
        runs_count = len(runs_paths)
        core_results = defaultdict(lambda: defaultdict(int))
        for run_name, run_path in runs_paths:
            for model_name, model_path in listdir(run_path):
                config_name = int(model_name.split('__')[0])
                core_results[config_name].update({ ('', k): v for k, v in configs[config_name].items() })
                for split_name, split_path in listdir(model_path):
                    # Get report files of each component
                    with open(os.path.join(split_path, 'story_report.json'), 'r') as f:
                        split_report = json.load(f)
                    core_results[config_name][(split_name, 'f1_action_prediction')] += (split_report['weighted avg']['f1-score'] * 100) / runs_count # Average over the runs
                    core_results[config_name][(split_name, 'story_accuracy')] += (split_report['conversation_accuracy']['accuracy'] * 100) / runs_count # Average over the runs
        core_results = pd.DataFrame.from_dict(core_results, orient='index').sort_index(axis=1, ascending=True).sort_values(('test', 'f1_action_prediction'), ascending=False)
        core_results.to_csv(os.path.join(work_dir, 'core_report.csv'), sep='\t', float_format='%.2f')
        

if __name__ == "__main__":
    args = parser.parse_args()
    exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_hyperopts(exp_name, args.n_iter, args.n_runs, args.percentages)
    process_results(exp_name)

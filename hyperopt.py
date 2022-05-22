"""Script to run hyperparameters optimization of a Rasa model."""
import os
import yaml
import tempfile
import argparse
import subprocess

from sklearn.model_selection import ParameterSampler

HYPERPARAMS = dict(
    epochs=[20, 50, 100, 200],
    max_history=[5, 10, 15],
    max_ngram=[3, 4, 5],
    fallback_threshold=[0.3, 0.5],
)

parser = argparse.ArgumentParser(description="Run hyperparameters optimization of a Rasa model.")
parser.add_argument('--n-iter', '-n', type=int, default=3, help="Total number of iterations to run.")

def set_hyperparams(config: dict, params: dict) -> dict:
    """Set the given hyperparams in the config dictionary."""
    if isinstance(config, dict):
        return { k: set_hyperparams(v, params) for k, v in config.items() } # Recursively set hyperparams in dict
    elif isinstance(config, list):
        return [ set_hyperparams(v, params) for v in config ] # Recursively set hyperparams in list
    elif isinstance(config, str) and config.startswith('$'):
        return params[config.lstrip('$')] # Set hyperparameter value
    return config



if __name__ == "__main__":
    args = parser.parse_args()
    # Load hyperopt config
    with open('config.hyperopt.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Generate config files to compare
    configs = []
    sampler = ParameterSampler(HYPERPARAMS, n_iter=args.n_iter, random_state=0)
    for i, params in enumerate(sampler):
        with tempfile.NamedTemporaryFile('w', delete=False) as tmp_config_file:
            # Generate new temp conig
            yaml.dump(set_hyperparams(config, params), tmp_config_file)
            configs.append(tmp_config_file.name)
    # Train and test NLU models with cross-validation
    print('Training and testing NLU models...')
    subprocess.call(['rasa', 'test', 'nlu',
        '--config', *configs,
        '--cross-validation',
        '--runs', '3',
        '--out', 'results/nlu',
        '--model', 'results/nlu/models'
    ])
    # Train and test dialogue models
    print('Training and testing dialogue models...')
    subprocess.call(['rasa', 'train', 'core',
        '--config', *configs,
        '--cross-validation',
        '--runs', '3',
        '--out', 'results/core/models'
    ])
    for split, stories_dir in dict(train='data', test='tests').items():  # The previous models have been trained excluding a certain amount of training data, so we can evaluate also over the train set
        subprocess.call(['rasa test core',
            '--model', 'results/core/models',
            '--stories', stories_dir,
            '--runs', '3',
            '--evaluate-model-directory'
            '--out', f'results/core/{split}'
        ])
    # Delete temp config files
    for config in configs:
        os.remove(config)

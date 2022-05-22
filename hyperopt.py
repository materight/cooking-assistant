"""Script to run hyperparameters optimization of a Rasa model."""
import os
import yaml
import tempfile
import argparse

from rasa.api import train, test
from sklearn.model_selection import ParameterSampler

HYPERPARAMS = dict(
    epochs=[20, 50, 100, 200],
    max_history=[5, 10, 15],
    max_ngram=[3, 4, 5],
    fallback_threshold=[0.3, 0.5],
)

parser = argparse.ArgumentParser(description="Run hyperparameters optimization of a Rasa model.")
parser.add_argument('--n-iter', '-n', type=int, default=50, help="Total number of iterations to run.")

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
    # Start hyperparamter search
    sampler = ParameterSampler(HYPERPARAMS, n_iter=args.n_iter, random_state=0)
    for i, params in enumerate(sampler):
        print(f'[{i+1}/{len(sampler)}] start training model with params:')
        print("\n".join(f"\t{k}: {v}" for k, v in params.items()))
        with tempfile.NamedTemporaryFile('w+', delete=False) as tmp_config_file:
            # Generate new temp conig
            tmp_config_dict = set_hyperparams(config, params)
            yaml.dump(tmp_config_dict, tmp_config_file)
            # Start training
            training_result = train(
                fixed_model_name='hyperopt',
                domain='domain.yml',
                config=tmp_config_file.name,
                training_files='data',
                force_training=False,
            )
            # Test trained model
            test(
                model='hyperopt'
            )
        os.unlink(tmp_config_file.name)
        print(training_result)
    # Save results
    os.makedirs(os.pat.join('results', 'hyperopt'), exist_ok=True)


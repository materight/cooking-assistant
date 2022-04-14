"""Simple script to generate data using chatette and convert it to the rasa YAML format."""
import os
import yaml
from chatette.facade import Facade as ChatetteFacade
from rasa.nlu.convert import convert_training_data

from actions.dataset import Dataset

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Generate NLU data with Chatette
print('Generating data with Chatette...')
chatette = ChatetteFacade(os.path.join(DATA_DIR, 'chatette', 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
chatette.run()

# Read Chatette output and convert it to the yaml format supported by rasa
print('Converting data to YAML format...')
convert_training_data(os.path.join(DATA_DIR, 'chatette', '.out'), os.path.join(DATA_DIR, 'rasa', 'nlu.yml'), 'yml', 'en')

# Generate base file for chatette with entity synonyms and lookup tables
print('Generating lookup tables and regex from dataset...')
dataset = Dataset()
with open(os.path.join(DATA_DIR, 'rasa', 'ingredients.yml'), 'w', encoding='utf-8') as file:
    yaml.dump(dict(version='3.1', nlu=[
        dict(lookup='ingredients', examples=dataset.ingredients)
    ]), file, sort_keys=False)

print('Done')

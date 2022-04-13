"""Simple script to generate data using chatette and convert it to the rasa YAML format."""
import os
from chatette.facade import Facade as ChatetteFacade
from rasa.nlu.convert import convert_training_data

from actions.dataset import Dataset

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Generate base file for chatette with entity synonyms and lookup tables
dataset = Dataset()


# Generate NLU data with Chatette
print('Generating data with Chatette...')
chatette = ChatetteFacade(os.path.join(DATA_DIR, 'chatette', 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
chatette.run()

# Read Chatette output and convert it to the yaml format supported by rasa
print('Converting data to YAML format...')
convert_training_data(os.path.join(DATA_DIR, 'chatette', '.out'), os.path.join(PROJECT_ROOT, 'data', 'rasa', 'nlu.yml'), 'yml', 'en')

print('Done')

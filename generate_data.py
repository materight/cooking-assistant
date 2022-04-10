"""Simple script to generate data using chatette and convert it to the rasa YAML format."""
import os
from chatette.facade import Facade as ChatetteFacade
from rasa.nlu.convert import convert_training_data

DATA_BASEPATH = os.path.join(os.path.dirname(__file__), 'data')
CHATETTE_DIR = os.path.join(DATA_BASEPATH, 'chatette')
OUTPUT_PATH = os.path.join(DATA_BASEPATH, 'rasa', 'nlu.yml')

print('Generating data with Chatette...')
chatette = ChatetteFacade(os.path.join(CHATETTE_DIR, 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
chatette.run()

print('Converting data to YAML format...')
convert_training_data(os.path.join(CHATETTE_DIR, '.out'), OUTPUT_PATH, 'yml', 'en')

print('Done')

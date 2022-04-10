"""Simple script to generate data using chatette and convert it to the rasa YAML format."""
import os
import re
import yaml
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from chatette.facade import Facade as ChatetteFacade
from rasa.nlu.convert import convert_training_data

RECIPES_URI = 'https://justthedarnrecipe.com'

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, '.cache')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Generate NLU data with Chatette
print('Generating data with Chatette...')
chatette = ChatetteFacade(os.path.join(DATA_DIR, 'chatette', 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
chatette.run()

# Read Chatette output and convert it to the yaml format supported by rasa
print('Converting data to YAML format...')
convert_training_data(os.path.join(DATA_DIR, 'chatette', '.out'), os.path.join(PROJECT_ROOT, 'data', 'rasa', 'nlu.yml'), 'yml', 'en')

print('Done')

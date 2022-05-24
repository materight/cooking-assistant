"""Simple script to generate data using chatette and convert it to the rasa YAML format."""
import os
import shutil
from typing import List, Text
from chatette.facade import Facade as ChatetteFacade
from rasa.nlu.convert import convert_training_data

from actions.dataset import Dataset

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def generate_entites_file(entity_name: Text, items: List[Text], file_path: Text, mode: Text = 'w'):
    """Generate a file containing a list of entities."""
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(f'\n@[{entity_name}]\n')
        for item in items:
            file.write(f'    {item}\n')


if __name__ == '__main__':
    # Init dataset
    dataset = Dataset()

    # Generate entities for recipes names
    print('Generating recipes keyword entities...')
    generate_entites_file('recipe', dataset.recipes, os.path.join(DATA_DIR, 'chatette', 'recipes.chatette'))

    # Generate entities for recipe tags
    print('Generating tags entities...')
    generate_entites_file('tag', dataset.tags, os.path.join(DATA_DIR, 'chatette', 'tags.chatette'))
    generate_entites_file('cuisine', dataset.cuisines, os.path.join(DATA_DIR, 'chatette', 'tags.chatette'), mode='a')

    # Generate entities for ingredient names
    print('Generating ingredients entities...')
    generate_entites_file('ingredient', dataset.ingredients, os.path.join(DATA_DIR, 'chatette', 'ingredients.chatette'))

    # Generate NLU data with Chatette
    print('Generating data with Chatette...')
    chatette = ChatetteFacade(os.path.join(DATA_DIR, 'chatette', 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
    chatette.run()

    # Read Chatette output and convert it to the yaml format supported by rasa
    print('Converting data to YAML format...')
    convert_training_data(os.path.join(DATA_DIR, 'chatette', '.out'), os.path.join(DATA_DIR, 'nlu.yml'), 'yml', 'en')
    shutil.rmtree(os.path.join(DATA_DIR, 'chatette', '.out'))

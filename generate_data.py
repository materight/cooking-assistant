"""Simple script to generate data using chatette and convert it to the rasa YAML format."""
import os
import shutil
from chatette.facade import Facade as ChatetteFacade
from rasa.nlu.convert import convert_training_data

from actions.dataset import Dataset

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

if __name__ == '__main__':
    # Init dataset
    dataset = Dataset()

    # Generate entities for ingredient names
    print('Generating ingredients entities...')
    with open(os.path.join(DATA_DIR, 'chatette', 'ingredients.chatette'), 'w', encoding='utf-8') as file:
        file.write('@[ingredient]\n')
        for ingredient in dataset.ingredients:
            file.write(f'    {ingredient}\n')

    # Generate entities for recipes names
    print('Generating recipes keyword entities...')
    with open(os.path.join(DATA_DIR, 'chatette', 'recipes.chatette'), 'w', encoding='utf-8') as file:
        file.write('@[recipe_keyword]\n')
        for recipe_name in dataset.recipes:
            file.write(f'    {recipe_name}\n')

    # Generate NLU data with Chatette
    print('Generating data with Chatette...')
    chatette = ChatetteFacade(os.path.join(DATA_DIR, 'chatette', 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
    chatette.run()

    # Read Chatette output and convert it to the yaml format supported by rasa
    print('Converting data to YAML format...')
    convert_training_data(os.path.join(DATA_DIR, 'chatette', '.out'), os.path.join(DATA_DIR, 'nlu.yml'), 'yml', 'en')
    shutil.rmtree(os.path.join(DATA_DIR, 'chatette', '.out'))

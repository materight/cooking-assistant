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

def process_time_str(time):
    time = time.replace('an', '1').replace('a', '1')
    if 'none' in time: return 0
    time, unit = time.split(' ')
    time = int(time)
    if 'hour' in unit: return time * 60
    return time

# Scrape recipes data if not present
recipes_path = os.path.join(DATA_DIR, 'recipes', 'recipes.yml')
if not os.path.exists(recipes_path):
    recipes = []
    for page in tqdm(range(1, 10)):
        resp = requests.get(f'{RECIPES_URI}/{page if page > 1 else ""}')
        resp.raise_for_status()
        html = BeautifulSoup(resp.text, 'html.parser')
        for a in html.select('div.MuiGrid-root div.MuiGrid-item a[href]'):
            resp = requests.get(f'{RECIPES_URI}/{a.attrs["href"]}')
            resp.raise_for_status()
            # Parse html to obtain attributes
            html = BeautifulSoup(resp.text, 'html.parser')
            title = html.select_one('div#main h1').text
            prep_time = html.select_one('div img[alt="prep time"] ~ div > h6').text
            cook_time = html.select_one('div img[alt="cook time"] ~ div > h6').text
            servings = html.select_one('div img[alt="servings"] ~ div > h6').text
            ingredients = [ i.text.strip() for i in html.select('div#main h1 ~ ul li, div#main h2 ~ ul li') ]
            steps = [ s.text.strip() for s in html.select('div#main h1 ~ ol li, div#main h2 ~ ol li') ]
            # Post-process
            title = title.lower()
            prep_time = process_time_str(prep_time) # Convert to minutes integer
            cook_time = process_time_str(cook_time)
            servings = int(servings)
            for i, ingr in enumerate(ingredients):
                amount = re.search(r'\d+( \d+)?(/\d+)?', ingr)
                unit = re.search(r'(oz|lbs?|tsps?|tbsp|jars?|cups?|packets?|teaspoons?|tablespoons?|pounds?)', ingr)
                if amount and not unit:
                    print('Not recognized', ingr)
                name = re.sub(r'\([^)]*\)', '', ingr)
                ingredients[i] = ingr # TODO: # dict(name=name, amount=amount)
            recipes.append(dict(title=title, prep_time=prep_time, cook_time=cook_time,
                                servings=servings, ingredients=ingredients, steps=steps))
    with open(os.path.join(DATA_DIR, 'recipes', 'recipes.yml'), 'w', encoding='utf-8') as file:
        yaml.dump(recipes, file, sort_keys=False, allow_unicode=True)

# Generate NLU data with Chatette
print('Generating data with Chatette...')
chatette = ChatetteFacade(os.path.join(DATA_DIR, 'chatette', 'main.chatette'), '.out', adapter_str='rasa', seed='0', force_overwriting=True, local=True)
chatette.run()

# Read Chatette output and convert it to the yaml format supported by rasa
print('Converting data to YAML format...')
convert_training_data(os.path.join(DATA_DIR, 'chatette', '.out'), os.path.join(PROJECT_ROOT, 'data', 'rasa', 'nlu.yml'), 'yml', 'en')

print('Done')

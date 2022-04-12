import os
import yaml
import pandas as pd
from typing import List, Text

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Dataset():
    """Dataset containing the recipes data used by the agent."""

    def __init__(self):
        # Load the data
        recipes_path = os.path.join(PROJECT_ROOT, 'data', 'recipes', 'recipes.yml')
        with open(recipes_path, 'r') as f:
            raw_recipes = yaml.load(f, Loader=yaml.FullLoader)
        # Convert to DataFrame
        self.recipes = pd.DataFrame([ dict(id=i, **r) for i, r in enumerate(raw_recipes)])
        self.ingredients = pd.DataFrame([ dict(recipe=i, **ingr) for i, r in enumerate(raw_recipes) for ingr in r['ingredients'] ])
        self.steps = pd.DataFrame([ dict(recipe=i, step=j, description=desc) for i, r in enumerate(raw_recipes) for j, desc in enumerate(r['steps']) ])
        # Post-processing
        self.recipes = self.recipes.drop(['ingredients', 'steps'], axis=1)
        self.ingredients[['amount', 'unit']] = self.ingredients.amount.fillna('').astype(str).str.split(r'(\d+)(.*)', expand=True)[[1,2]].replace('', None)
        self.ingredients['amount'] = self.ingredients['amount'].astype(float)

    def search_by_title(self, query: Text):
        """Search for a recipe by title."""
        results = self.recipes[self.recipes['title'].str.contains(query, case=False)]
        return results

    def search_by_ingredients(self, query: List[Text]):
        """Search for a recipe by ingredients."""
        query = '|'.join(query)
        results = self.ingredients[self.ingredients['name'].str.contains(query)]
        # Count number of ingredients occurences
        results = results.groupby('recipe', as_index=False).name.count().rename(columns={'name': 'count'})
        results = results.sort_values('count', ascending=False)
        return results

d = Dataset()
print('ok')

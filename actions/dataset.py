
import os
import yaml
import pandas as pd
from dataclasses import dataclass
from typing import List, Text, Optional

from . import utils


PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

@dataclass
class Ingredient:
    recipe_id: int
    name: Text
    amount: Optional[float]
    unit: Optional[Text]
    def __str__(self) -> str:
        return utils.ingredient_to_str(self.name, self.amount, self.unit)

@dataclass
class Step:
    recipe_id: int
    step_index: int
    description: Text
    def __str__(self) -> str:
        return self.description

@dataclass
class Recipe:
    id: int
    title: Text
    image: Optional[Text]
    prep_time: int
    cook_time: int
    servings: int
    ingredients: List[Ingredient]
    steps: List[Step]

    def set_servings(self, servings: int):
        for ingredient in self.ingredients:
            ingredient.amount = ingredient.amount * (servings / self.servings)
        self.servings = servings



class Dataset():
    """Dataset containing the recipes data used by the agent."""

    def __init__(self):
        # Load the data
        recipes_path = os.path.join(PROJECT_ROOT, 'data', 'recipes', 'recipes.yml')
        with open(recipes_path, 'r', encoding='utf-8') as f:
            raw_recipes = yaml.load(f, Loader=yaml.FullLoader)
        ingredients_substitutes_path = os.path.join(PROJECT_ROOT, 'data', 'recipes', 'ingredients_substitutes.yml')
        with open(ingredients_substitutes_path, 'r', encoding='utf-8') as f:
            raw_ingredients_substitutes = yaml.load(f, Loader=yaml.FullLoader)
        # Convert to DataFrame
        self._df_recipes = pd.DataFrame([ dict(id=i, **r) for i, r in enumerate(raw_recipes)])
        self._df_ingredients = pd.DataFrame([ dict(recipe_id=i, **ingr) for i, r in enumerate(raw_recipes) for ingr in r['ingredients'] ])
        self._df_steps = pd.DataFrame([ dict(recipe_id=i, step_index=j, description=desc) for i, r in enumerate(raw_recipes) for j, desc in enumerate(r['steps']) ])
        self._df_ingredients_substitutes = pd.DataFrame([next(iter(i.items())) for i  in raw_ingredients_substitutes], columns=['name', 'substitute'])
        # Post-processing
        self._df_recipes = self._df_recipes.drop(['ingredients', 'steps'], axis=1).set_index('id')
        self._df_ingredients[['amount', 'unit']] = self._df_ingredients.amount.fillna('').astype(str).str.split(r'(\d+)(.*)', expand=True)[[1,2]]
        self._df_ingredients['amount'] = self._df_ingredients['amount'].astype(float)

    @property
    def recipes(self) -> List[Text]:
        """Returns a list of all the available recipes titles."""
        return self._df_recipes.title.unique().tolist()

    @property
    def ingredients(self) -> List[Text]:
        """Returns a list of all the available ingredients."""
        return self._df_ingredients.name.unique().tolist()

    def get_recipe(self, recipe_id: int) -> Recipe:
        """Converts a recipe id to the corresponding Recipe objects."""
        df_recipe = self._df_recipes.loc[recipe_id]
        df_ingredients = self._df_ingredients[self._df_ingredients.recipe_id == recipe_id]
        df_steps = self._df_steps[self._df_steps.recipe_id == recipe_id].sort_values(by='step_index')
        ingredients = [ Ingredient(**ingr) for ingr in df_ingredients.to_dict('records') ]
        steps = [ Step(**step) for step in df_steps.to_dict('records') ]
        recipe = Recipe(id=recipe_id, **df_recipe, ingredients=ingredients, steps=steps)
        return recipe

    def search_by_keyword(self, query: Text) -> List[int]:
        """Search for a recipe by keyword."""
        recipes_ids = self._df_recipes[self._df_recipes['title'].str.contains(query, case=False)].index.to_list()
        return recipes_ids

    def search_by_ingredients(self, query: List[Text]) -> List[int]:
        """Search for a recipe by ingredients."""
        query = '|'.join(query)
        results = self._df_ingredients[self._df_ingredients['name'].str.contains(query, case=False)]
        # Count number of ingredients occurences
        results = results.groupby('recipe_id', as_index=False).name.count().rename(columns={'name': 'count'})
        recipes_ids = results.sort_values('count', ascending=False).recipe_id.to_list() # Returns list ordered by number of mathched ingredients
        return recipes_ids

    def search_ingredient_substitute(self, query: Text) -> Optional[Text]:
        """Search for an alternative to the given ingredient."""
        results = self._df_ingredients_substitutes[self._df_ingredients_substitutes['name'].str.contains(query, case=False)].substitute.tolist()
        if len(results) == 0: 
            return None
        return results[0]

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
    tags: List[Text]
    cuisine: Optional[Text]
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
        return sorted(self._df_recipes.title.unique().tolist())

    @property
    def ingredients(self) -> List[Text]:
        """Returns a list of all the available ingredients."""
        return sorted(self._df_ingredients.name.unique().tolist())

    @property
    def tags(self) -> List[Text]:
        """Returns a list of all the available tags."""
        return sorted(self._df_recipes.tags.explode().dropna().unique().tolist())

    @property
    def cuisines(self) -> List[Text]:
        """Returns a list of all the available cuisines."""
        return sorted(self._df_recipes.cuisine.dropna().unique().tolist())

    def get_recipe(self, recipe_id: int) -> Recipe:
        """Converts a recipe id to the corresponding Recipe objects."""
        df_recipe = self._df_recipes.loc[recipe_id]
        df_ingredients = self._df_ingredients[self._df_ingredients.recipe_id == recipe_id]
        df_steps = self._df_steps[self._df_steps.recipe_id == recipe_id].sort_values(by='step_index')
        ingredients = [ Ingredient(**ingr) for ingr in df_ingredients.to_dict('records') ]
        steps = [ Step(**step) for step in df_steps.to_dict('records') ]
        recipe = Recipe(id=recipe_id, **df_recipe, ingredients=ingredients, steps=steps)
        return recipe

    def search_recipes(self, keywords: List[Text], ingredients: List[Text], tags: List[Text], cuisine: Optional[Text]) -> List[int]:
        """Search for recipes matching the given keywords, ingredients, tags and cuisine."""
        # Pre-processing
        tags = set(t.lower() for t in tags)
        keywords = [ k for k in keywords if k not in tags and k != cuisine ]
        ingredients = [ i for i in ingredients if i not in tags and i != cuisine ]
        # Search with filters
        recipes_mask = True
        if len(keywords) > 0:
            recipes_mask &= self._df_recipes['title'].str.contains('|'.join(keywords), case=False) # Any of the keywords
        if len(ingredients) > 0:
            results = self._df_ingredients[self._df_ingredients['name'].str.contains('|'.join(ingredients), case=False)] # All of the ingredients
            results = results.groupby('recipe_id', as_index=False).name.count().rename(columns=dict(name='count')) # Count number of ingredients occurences
            results = results[results['count'] == len(ingredients)].recipe_id.to_list() # Returns only the recipes containing all the given ingredients
            recipes_mask &= self._df_recipes.index.isin(results)
        if len(tags) > 0:
            recipes_mask &= self._df_recipes['tags'].apply(tags.issubset)
        if cuisine is not None:
            recipes_mask &= self._df_recipes['cuisine'].str.contains(cuisine, case=False)
        recipe_ids = self._df_recipes[recipes_mask].index.tolist()
        return recipe_ids

    def search_ingredient_substitute(self, query: Text) -> Optional[Text]:
        """Search for an alternative to the given ingredient."""
        results = self._df_ingredients_substitutes[self._df_ingredients_substitutes['name'].str.contains(query, case=False)].substitute.tolist()
        if len(results) == 0: 
            return None
        return results[0]

"""Custom actions."""
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from .dataset import Dataset, Ingredient

# Load dataset globally
dataset = Dataset()


class ActionSearchByIngredients(Action):
    """Search for a recipe by ingredients."""

    def name(self) -> Text:
        return 'action_search_by_ingredients'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ingredient = next(tracker.get_latest_entity_values('ingredient'), None)
        print(ingredient)
        
        recipes_ids = dataset.search_by_ingredients([ingredient])

        if len(recipes_ids) == 0:
            dispatcher.utter_message(response='utter_recipe_not_found')
        else:
            recipe = dataset.get_recipe(recipes_ids[0])
            print(recipe.title)
            dispatcher.utter_message(response='utter_recipe_found', recipe_title=recipe.title)
            return []# [SlotSet('found_recipes', recipe_results)]




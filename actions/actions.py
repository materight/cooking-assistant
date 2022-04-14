"""Custom actions."""
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from .dataset import Dataset

# Load dataset globally
dataset = Dataset()

class ActionSearchByTitle(Action):
    """Search for a recipe by title."""

    def name(self) -> Text:
        return 'action_search_by_keyword'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipe_keyword = tracker.get_latest_entity_values('recipe_keyword', None)
        
        recipe_results = dataset.search_by_keyword(recipe_keyword)

        if len(recipe_results) == 0:
            dispatcher.utter_message(template='utter_recipe_not_found')
        else:
            dispatcher.utter_message(template='utter_recipe_found')
            return [SlotSet('found_recipes', recipe_results)]




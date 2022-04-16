"""Custom actions."""
import logging
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from .dataset import Dataset, Ingredient

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load dataset globally
dataset = Dataset()

class ActionSearchByIngredients(Action):
    """Search for a recipe by ingredients."""

    def name(self) -> Text:
        return 'action_search_by_ingredients'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ingredients = list(tracker.get_latest_entity_values('ingredient'))
        logger.info('Search recipe for ingredients %s', ingredients)
        if len(ingredients) == 0:
            dispatcher.utter_message(response='utter_search_recipe/not_found')
            return []
        recipes_ids = dataset.search_by_ingredients(ingredients)
        logger.info('Found %d recipes',len(recipes_ids))
        if len(recipes_ids) == 0:
            dispatcher.utter_message(response='utter_search_recipe/not_found')
            return []
        else:
            recipe = dataset.get_recipe(recipes_ids[0])
            dispatcher.utter_message(response='utter_search_recipe/found', recipe_title=recipe.title)
            return [ SlotSet('found_recipes_ids', recipes_ids), SlotSet('current_recipe_id', recipes_ids[0]) ]


class ActionListIngredients(Action):
    """List all the ingredients needed for the selected recipe."""

    def name(self) -> Text:
        return 'action_list_ingredients'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_recipe_id = tracker.get_slot('current_recipe_id')
        if current_recipe_id is not None:
            logger.info('Listing ingredients for recipe %s', current_recipe_id)
            recipe = dataset.get_recipe(current_recipe_id)
            ingredients_list = '\n'.join([ f'  - {i}' for i in recipe.ingredients ])
            dispatcher.utter_message(response='utter_list_ingredients', ingredients_list=ingredients_list)
        else:
            dispatcher.utter_message(response='utter_search_recipe/not_found')
        return []


class ActionListStepsLoop(FormValidationAction):
    """Form validator to read step-by-step the instructions for a recipe."""

    def name(self) -> Text:
        return 'validate_list_steps_loop'

    def validate_list_steps_done(self, value, dispatcher, tracker, domain):
        current_recipe_id = tracker.get_slot('current_recipe_id')
        current_step_idx = tracker.get_slot('current_step_idx')
        current_step_idx += 1 # Go to the next step
        recipe = dataset.get_recipe(current_recipe_id)
        if current_step_idx >= len(recipe.steps):
            # All the steps have been read
            dispatcher.utter_message(response='utter_list_steps/end', step_description=current_step_descr)
            return dict(current_step_idx=-1, list_steps_done=True)
        else:
            # Read next step
            current_step_descr = recipe.steps[current_step_idx].description
            if current_step_idx == 0:
                dispatcher.utter_message(response='utter_list_steps/first', step_description=current_step_descr)
            else:
                dispatcher.utter_message(response='utter_list_steps/next', step_description=current_step_descr)
            return dict(current_step_idx=current_step_idx, list_steps_done=None)

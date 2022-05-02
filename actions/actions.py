"""Custom actions."""
import logging
from typing import Any, Text, Dict, List

from word2number import w2n
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from .dataset import Dataset, Ingredient, Recipe

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
            recipe = dataset.get_recipe(recipes_ids[0]) # Return first recipe
            dispatcher.utter_message(response='utter_search_recipe/found', recipe_title=recipe.title)
            return [ SlotSet('found_recipes_ids', recipes_ids), SlotSet('current_recipe', recipe.id) ]


class ActionSearchAlternativeRecipe(Action):
    """Give the user an alternative to the selected recipe."""
    def name(self) -> Text:
        return 'action_search_alternative_recipe'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        current_recipe_id = tracker.get_slot('current_recipe') # TODO: handle None recipe
        found_recipes_ids = tracker.get_slot('found_recipes_ids') # TODO: handle None recipe_ids
        if len(found_recipes_ids) <= 1:
            dispatcher.utter_message(response='utter_search_recipe/not_found_alternative')
            return []
        current_recipe_idx = found_recipes_ids.index(current_recipe_id)
        new_recipe_id = found_recipes_ids[(current_recipe_idx + 1) % len(found_recipes_ids)]
        recipe = dataset.get_recipe(new_recipe_id)
        dispatcher.utter_message(response='utter_search_recipe/found_alternative', recipe_title=recipe.title)
        return [ SlotSet('current_recipe', new_recipe_id) ]
        

class ActionTellExpectedTime(Action):
    """Tell the user the expected preparation and cooking time."""

    def name(self) -> Text:
        return 'action_tell_expected_time'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipe_id = tracker.get_slot('current_recipe') # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        dispatcher.utter_message(response='utter_expected_time', prep_time=str(recipe.prep_time), cook_time=str(recipe.cook_time))
        return []


class ActionListIngredients(Action):
    """List all the ingredients needed for the selected recipe."""

    def name(self) -> Text:
        return 'action_list_ingredients'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipe_id = tracker.get_slot('current_recipe')  # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        people_count = tracker.get_slot('people_count')
        logger.info('Found %d ingredients', len(recipe.ingredients))
        if people_count is None:
            logger.info('Use default recipe servings: %d people', recipe.servings)
            people_count = recipe.servings  # Use recipe's servings as people_count value
        else:
            # Update ingredients amount to adapt to the specified people_count
            people_count = w2n.word_to_num(people_count)
            logger.info('Update recipe to adapt to %d people', people_count)
            for i in recipe.ingredients:
                i.amount = i.amount * (people_count / recipe.servings)
            recipe.servings = people_count
        ingredients_list = '\n'.join([ f'  - {i}' for i in recipe.ingredients ])
        people_count_str = f'{people_count} people' if people_count > 1 else '1 person'
        dispatcher.utter_message(response='utter_list_ingredients', ingredients_list=ingredients_list, people_count_str=people_count_str)
        return []


class ActionSearchIngredientSubstitute(Action):
    """Search for an alternative to the given ingredient."""

    def name(self) -> Text:
        return 'action_search_ingredient_substitute'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ingredients = list(tracker.get_latest_entity_values('ingredient')) # TODO: handle multiple ingredients
        ingredient, substitute = None, None
        if len(ingredients) > 0:
            ingredient = ingredients[0]
            substitute = dataset.search_ingredient_substitute(ingredient)
            logger.info('Substitute for ingredient "%s": %s', ingredient, substitute)        
        if substitute is not None:
            dispatcher.utter_message(response='utter_ingredient_substitute/found', ingredient=ingredient, substitute=substitute)
        elif ingredient is not None:
            dispatcher.utter_message(response='utter_ingredient_substitute/not_found', ingredient=ingredient)
        else:
            dispatcher.utter_message(response='utter_ingredient_substitute/no_ingredient')
        return []


class ActionListStepsLoop(FormValidationAction):
    """Form validator to read step-by-step the instructions for a recipe."""

    def name(self) -> Text:
        return 'validate_list_steps_loop'

    def validate_list_steps_done(self, value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any])-> Dict[Text, Any]:
        recipe_id = tracker.get_slot('current_recipe') # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        current_step_idx = tracker.get_slot('current_step_idx')
        current_step_idx += 1 # Go to the next step
        logger.info('Reading step %d/%d of recipe %s', current_step_idx + 1, len(recipe.steps), recipe.id)
        if current_step_idx >= len(recipe.steps):
            # All the steps have been read
            dispatcher.utter_message(response='utter_list_steps/end')
            return dict(current_step_idx=-1, list_steps_done=True)
        else:
            # Read next step
            current_step_descr = recipe.steps[current_step_idx].description
            current_step_descr = current_step_descr[0].lower() + current_step_descr[1:]
            if current_step_idx == 0:
                dispatcher.utter_message(response='utter_list_steps/first', step_description=current_step_descr)
            else:
                dispatcher.utter_message(response='utter_list_steps/next', step_description=current_step_descr)
            return dict(current_step_idx=current_step_idx, list_steps_done=None)

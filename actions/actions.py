"""Custom Rasa actions."""
import logging
from datetime import datetime, timedelta
from typing import Any, Text, Dict, List

from word2number import w2n
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet, AllSlotsReset, ReminderScheduled
from rasa_sdk.executor import CollectingDispatcher

from . import utils
from .dataset import Dataset

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load dataset globally
dataset = Dataset()

class ActionSearchByKeyword(Action):
    """Search for a recipe by keyword."""

    def name(self) -> Text:
        return 'action_search_by_keyword'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        keyword = next(tracker.get_latest_entity_values('recipe_keyword'), None)
        logger.info('Search recipe by keyword "%s"', keyword)
        if keyword is None:
            dispatcher.utter_message(response='utter_search_recipe/not_found')
            return []
        recipes_ids = dataset.search_by_keyword(keyword)
        logger.info('Found %d recipes',len(recipes_ids))
        if len(recipes_ids) == 0:
            dispatcher.utter_message(response='utter_search_recipe/not_found')
            return []
        else:
            recipe = dataset.get_recipe(recipes_ids[0]) # Return first recipe
            dispatcher.utter_message(response='utter_search_recipe/found', recipe_title=recipe.title, image=recipe.image)
            return [ SlotSet('found_recipes_ids', recipes_ids), SlotSet('current_recipe', recipe.id) ]


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


class ActionUpdatePeopleCount(Action):
    def name(self) -> Text:
        return 'action_update_people_count'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message['intent'].get('name')
        if intent == 'tell_people_count_one':
            logger.info('Set people count to 1')
            return [ SlotSet('people_count', str(1)) ]
        people_count_str = next(tracker.get_latest_entity_values('CARDINAL'), None)
        if people_count_str is not None:
            logger.info('Set people count to %s', people_count_str)
            return [ SlotSet('people_count', people_count_str) ]
        logger.info('No people count found')
        return []


class ActionListIngredients(Action):
    """List all the ingredients needed for the selected recipe."""

    def name(self) -> Text:
        return 'action_list_ingredients'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipe_id = tracker.get_slot('current_recipe')  # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        people_count = next(tracker.get_latest_entity_values('CARDINAL'), 
                            tracker.get_slot('people_count')) # Use value o entity or current slot as fallback
        logger.info('Listing ingredients for recipe %s and "%s" people', recipe.id, people_count)
        logger.info('Found %d ingredients', len(recipe.ingredients))
        if people_count is None:
            logger.info('Use default recipe servings: %d people', recipe.servings)
            people_count = recipe.servings  # Use recipe's servings as people_count value
        else:
            # Update ingredients amount to adapt to the specified people_count
            people_count = w2n.word_to_num(str(people_count))
            logger.info('Update recipe to adapt to %d people', people_count)
            recipe.set_servings(people_count)
        ingredients_list = '\n'.join([ f'  - {i}' for i in recipe.ingredients ])
        people_count_str = f'{people_count} people' if people_count > 1 else '1 person'
        dispatcher.utter_message(response='utter_list_ingredients', ingredients_list=ingredients_list, people_count_str=people_count_str)
        return []


class ActionSearchIngredientSubstitute(Action):
    """Search for an alternative to the given ingredient."""

    def name(self) -> Text:
        return 'action_search_ingredient_substitute'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ingredients = list(tracker.get_latest_entity_values('ingredient')) # TODO: handle multiple ingredients and None case
        ingredient, substitute = None, None
        if len(ingredients) > 0:
            ingredient = ingredients[0]
            substitute = dataset.search_ingredient_substitute(ingredient)
            logger.info('Substitute for ingredient "%s": %s', ingredient, substitute)        
        if substitute is not None:
            dispatcher.utter_message(response='utter_ingredient_substitute/found', substitute=substitute)
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
        if current_step_idx >= len(recipe.steps):
            # All the steps have been read
            logger.info('All the steps of recipe %s have been read', recipe.id)
            dispatcher.utter_message(response='utter_list_steps/end')
            return dict(current_step_idx=-1, list_steps_done=True)
        else:
            # Read next step
            logger.info('Reading step %d/%d of recipe %s', current_step_idx + 1, len(recipe.steps), recipe.id)
            current_step_descr = utils.lower_first_letter(recipe.steps[current_step_idx].description)
            if current_step_idx == 0:
                dispatcher.utter_message(response='utter_list_steps/first', step_description=current_step_descr)
            else:
                dispatcher.utter_message(response='utter_list_steps/next', step_description=current_step_descr)
            return dict(current_step_idx=current_step_idx, list_steps_done=None)


class ActionSetTimer(Action):
    """Set a timer as reminder."""
    
    def name(self) -> Text:
        return 'action_set_timer'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        time_str = next(tracker.get_latest_entity_values('TIME'), None)  #TODO: handle None case
        if time_str is not None:
            amount, unit = utils.parse_time_str(time_str)
            if amount is not None and unit is not None:
                trigger_time = datetime.now() + timedelta(**{unit: amount})
                logger.info('Set a timer for %d %s, trigger at %s', amount, unit, trigger_time)
                dispatcher.utter_message(response='utter_set_timer/done', time=f'{amount} {unit}')
                return [ ReminderScheduled(trigger_date_time=trigger_time, intent_name='EXTERNAL_timer_expired', kill_on_user_message=False) ]
        logger.info('Could not set timer for entity "%s"', time_str)
        dispatcher.utter_message(response='utter_set_timer/error', time=time_str)
        return [ ]


class ActionRepeatLastUtterance(Action):
    """Repeat the last utterance sent to the user."""

    def name(self) -> Text:
        return "action_repeat_last_utterance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        texts = []
        for event in reversed(tracker.events):
            if event.get('event') != 'bot': # Get utterances until a user message is found
                break
            texts.append(event.get('text'))
        for text in reversed(texts):
            dispatcher.utter_message(text=text)
        return []

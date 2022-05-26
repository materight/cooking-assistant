"""Custom Rasa actions."""
import logging
from datetime import datetime, timedelta
from typing import Any, Text, Dict, List

from word2number import w2n
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet, ReminderScheduled, FollowupAction
from rasa_sdk.executor import CollectingDispatcher

from . import utils
from .dataset import Dataset

# Init logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load dataset globally
dataset = Dataset()


class ActionSearchRecipe(Action):
    """Search for a recipe by keyword, ingredients, tags or cuisine."""
    
    def name(self) -> Text:
        return 'action_search_recipes'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        keywords = list(tracker.get_latest_entity_values('recipe'))
        ingredients = list(tracker.get_latest_entity_values('ingredient'))
        tags = list(tracker.get_latest_entity_values('tag'))
        cuisine = next(tracker.get_latest_entity_values('cuisine'), None)
        logger.info('Search recipe by keywords %s, ingredients %s, tags %s and cuisine "%s"', keywords, ingredients, tags, cuisine)
        if len(keywords) == 0 and len(ingredients) == 0 and len(tags) == 0 and cuisine is None:
            dispatcher.utter_message(response='utter_search_recipe_not_found')
            return []
        recipes_ids = dataset.search_recipes(keywords, ingredients, tags, cuisine)
        logger.info('Found %d recipes', len(recipes_ids))
        if len(recipes_ids) == 0:
            dispatcher.utter_message(response='utter_search_recipe_not_found')
            return []
        elif len(recipes_ids) == 1:  # Return the single recipe found
            recipe = dataset.get_recipe(recipes_ids[0])
            dispatcher.utter_message(response='utter_search_recipe_found', recipe_title=recipe.title, image=recipe.image)
            return [ SlotSet('found_recipes_ids', recipes_ids), SlotSet('current_recipe_id', recipe.id) ]
        else: # More alternatives found, asks the user for more details
            return [ SlotSet('found_recipes_ids', recipes_ids), FollowupAction('action_refine_recipes_search') ]


class ActionRefineRecipesSearchAsk(Action):
    """Asks more questions to narrow down the found recipes."""
        
    def name(self) -> Text:
        return 'action_refine_recipes_search_ask'
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipes_ids = tracker.get_slot('found_recipes_ids')
        prop, value = dataset.get_discriminative_properties(recipes_ids)
        if prop is not None: # Ask the user for more details
            dispatcher.utter_message(response='utter_refine_recipes_search', value=value)
            return [ SlotSet('refine_recipes_search_prop', prop), SlotSet('refine_recipes_search_value', value) ]
        else:  # If not discriminative property was found, return the first recipe
            recipe = dataset.get_recipe(recipes_ids[0])
            dispatcher.utter_message(response='utter_search_recipe_found', recipe_title=recipe.title, image=recipe.image)
            return [ SlotSet('found_recipes_ids', recipes_ids), SlotSet('current_recipe_id', recipe.id) ]


class ActionRefineRecipesSearchFilter(Action):
    """Filter the found recipes by the user's input."""
        
    def name(self) -> Text:
        return 'action_refine_recipes_search_filter'
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        found_recipes_ids = tracker.get_slot('found_recipes_ids')
        prop, value = tracker.get_slot('refine_recipes_search_prop'), tracker.get_slot('refine_recipes_search_value')
        # Filter the found recipes according to the user's positive or negative response. In case 'idk' is received, do not filter the recipes.
        user_response = tracker.latest_message['intent'].get('name')
        if user_response == 'affirm':
            recipes_ids = dataset.filter_recipes_by_property(found_recipes_ids, prop, value, negative=False)
        elif user_response == 'deny':
            recipes_ids = dataset.filter_recipes_by_property(found_recipes_ids, prop, value, negative=True)
        # Return the first of the filtered recipes
        recipe = dataset.get_recipe(recipes_ids[0])
        dispatcher.utter_message(response='utter_search_recipe_found', recipe_title=recipe.title, image=recipe.image)
        return [ SlotSet('found_recipes_ids', recipes_ids), SlotSet('current_recipe_id', recipe.id), 
                 SlotSet('refine_recipes_search_prop', None), SlotSet('refine_recipes_search_value', None) ]
        # TODO: ask again other questions


class ActionSearchAlternativeRecipe(Action):
    """Give the user an alternative to the selected recipe."""

    def name(self) -> Text:
        return 'action_search_alternative_recipe'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        found_recipes_ids = tracker.get_slot('found_recipes_ids')
        current_recipe_id = tracker.get_slot('current_recipe_id') # TODO: handle None recipe
        if found_recipes_ids is None or len(found_recipes_ids) <= 1:
            dispatcher.utter_message(response='utter_search_recipe_not_found_alternative')
            return []
        current_recipe_idx = found_recipes_ids.index(current_recipe_id)
        new_recipe_id = found_recipes_ids[(current_recipe_idx + 1) % len(found_recipes_ids)]
        recipe = dataset.get_recipe(new_recipe_id)
        dispatcher.utter_message(response='utter_search_recipe_found_alternative', recipe_title=recipe.title)
        return [ SlotSet('current_recipe_id', new_recipe_id) ]
        

class ActionTellExpectedTime(Action):
    """Tell the user the expected preparation and cooking time."""

    def name(self) -> Text:
        return 'action_tell_expected_time'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipe_id = tracker.get_slot('current_recipe_id') # TODO: handle None recipe
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
        recipe_id = tracker.get_slot('current_recipe_id')  # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        people_count = next(tracker.get_latest_entity_values('CARDINAL'), tracker.get_slot('people_count')) # Use value o entity or current slot as fallback
        logger.info('Listing ingredients for recipe %s and "%s" people, found %d ingredients', recipe.id, people_count, len(recipe.ingredients))
        if people_count is None:
            logger.info('Use default recipe servings: %d people', recipe.servings)
            people_count = recipe.servings  # Use recipe's servings as people_count value
        else:
            # Update ingredients amount to adapt to the specified people_count
            people_count = w2n.word_to_num(str(people_count))
            logger.info('Update recipe to adapt to %d people', people_count)
            recipe.set_servings(people_count)
        ingredients_list = '\n'.join([ f'  - {ingredient}' for ingredient in recipe.ingredients ])
        people_count_str = f'{people_count} people' if people_count > 1 else '1 person'
        dispatcher.utter_message(response='utter_list_ingredients', ingredients_list=ingredients_list, people_count_str=people_count_str)
        return []


class ActionSearchIngredientsSubstitutes(Action):
    """Search for alternatives to the given ingredients."""

    def name(self) -> Text:
        return 'action_search_ingredients_substitutes'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        ingredients = list(tracker.get_latest_entity_values('ingredient'))
        if len(ingredients) == 0:
            dispatcher.utter_message(response='utter_ingredient_substitute_no_ingredient')
            return []
        # Search for substitutes
        substitutes = dataset.search_ingredients_substitutes(ingredients)
        logger.info('Substitute for ingredients %s = %s', ingredients, substitutes)
        # Utter substitutes
        if len(substitutes) == 0:
            ingredients_str = utils.join_list_str(ingredients, last_sep='or')
            dispatcher.utter_message(response='utter_ingredient_substitute_not_found', ingredient=ingredients_str)
        else:
            substitutes_str = utils.join_list_str(substitutes, last_sep='and')
            dispatcher.utter_message(response='utter_ingredient_substitute_found', substitute=substitutes_str)
        return []


class ActionTellIngredientAmount(Action):
    def name(self) -> Text:
        return 'action_tell_ingredient_amount'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        recipe_id = tracker.get_slot('current_recipe_id')  # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        people_count = next(tracker.get_latest_entity_values('CARDINAL'), tracker.get_slot('people_count'))  # Use value of entity or current slot as fallback
        asked_ingredients = list(tracker.get_latest_entity_values('ingredient'))
        if len(asked_ingredients) > 0:
            if people_count is not None: # Update ingredients amount to adapt to the specified people_count
                recipe.set_servings(w2n.word_to_num(str(people_count)))
            amounts = [ utils.ingredient_to_str(ingr.name, ingr.amount, ingr.unit, default_amount='some') for ingr in recipe.ingredients 
                        if any(asked_ingr in ingr.name for asked_ingr in asked_ingredients) ]
            if len(amounts) > 0:
                amounts_str = utils.join_list_str(amounts)
                dispatcher.utter_message(response='utter_ingredient_amount_found', amounts_str=amounts_str)
            else:
                ingredients_str = utils.join_list_str(asked_ingredients, last_sep='or')
                dispatcher.utter_message(response='utter_ingredient_amount_not_found', ingredients_str=ingredients_str)
        else:
            dispatcher.utter_message(response='utter_ingredient_amount_no_ingredient')
        return []


class ActionListStepsLoop(FormValidationAction):
    """Form validator to read step-by-step the instructions for a recipe."""

    def name(self) -> Text:
        return 'validate_list_steps_loop'

    def validate_list_steps_done(self, value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any])-> Dict[Text, Any]:
        recipe_id = tracker.get_slot('current_recipe_id') # TODO: handle None recipe
        recipe = dataset.get_recipe(recipe_id)
        current_step_idx = tracker.get_slot('current_step_idx')
        current_step_idx += 1 # Go to the next step
        if current_step_idx >= len(recipe.steps):
            # All the steps have been read
            logger.info('All the steps of recipe %s have been read', recipe.id)
            dispatcher.utter_message(response='utter_list_steps_end')
            return dict(current_step_idx=-1, list_steps_done=True)
        else:
            # Read next step
            logger.info('Reading step %d/%d of recipe %s', current_step_idx + 1, len(recipe.steps), recipe.id)
            current_step_descr = utils.lower_first_letter(recipe.steps[current_step_idx].description)
            if current_step_idx == 0:
                dispatcher.utter_message(response='utter_list_steps_first', step_description=current_step_descr)
            else:
                dispatcher.utter_message(response='utter_list_steps_next', step_description=current_step_descr)
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
                dispatcher.utter_message(response='utter_set_timer_done', time=f'{amount} {unit}')
                return [ ReminderScheduled(trigger_date_time=trigger_time, intent_name='EXTERNAL_timer_expired', kill_on_user_message=False) ]
        logger.info('Could not set timer for entity "%s"', time_str)
        dispatcher.utter_message(response='utter_set_timer_error', time=time_str)
        return [ ]


class ActionRepeatLastUtterance(Action):
    """Repeat the last utterance sent to the user."""

    def name(self) -> Text:
        return "action_repeat_last_utterance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logger.info(f'Repeating last utterance')
        for event in reversed(tracker.events):
            if event.get('event') == 'bot': # Get utterances until a user message is found
                dispatcher.utter_message(text=event.get('text'))
                break
        return []


class ActionResetListStepsLoop(Action):
    """Reset the slots for reading the steps of a recipe."""

    def name(self) -> Text:
        return 'action_reset_list_steps_loop'

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logger.info('Resetting the list_steps_loop slots')
        return [ SlotSet('current_step_idx', -1), SlotSet('list_steps_done', None) ]


"""Custom actions."""
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Load dataset globally
dataset = Dataset()

class ActionSearchByTitle(Action):
    """Search for a recipe by title."""

    def name(self) -> Text:
        return "action_search_by_title"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Hello World!")
        return []



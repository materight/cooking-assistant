"""Google Assistant connector for Rasa"""
import logging
from sanic import Blueprint, response

from rasa.core.channels.channel import UserMessage
from rasa.core.channels.channel import InputChannel
from rasa.core.channels.channel import CollectingOutputChannel

logger = logging.getLogger(__name__)

class GoogleConnector(InputChannel):
    """Custom connector to Google Assistant."""

    @classmethod
    def name(cls):
        return 'google_assistant'

    def blueprint(self, on_new_message):
        google_webhook = Blueprint('google_webhook', __name__)

        @google_webhook.route('/', methods=['GET'])
        async def health(request):
            return response.json({'status': 'ok'})

        @google_webhook.route('/webhook', methods=['POST'])
        async def receive(request):
            payload = request.json	
            intent = payload['inputs'][0]['intent'] 			
            text = payload['inputs'][0]['rawInputs'][0]['query'] 
	
            items = []
            if intent == 'actions.intent.MAIN':
                items.append({'simpleResponse': { 'textToSpeech': 'Hello! Welcome to the Cooking assistant! What would you like to prepare today?'	}})
            else:
                out = CollectingOutputChannel()
                await on_new_message(UserMessage(text, out))
                logger.info("Received message: %s", out.messages)
                for m in out.messages:
                    if 'text' in m:
                        items.append({'simpleResponse': { 'textToSpeech': m['text'] }})
                    elif 'image' in m:
                        items.append({'basicCard': { 'image': { 'url': m['image'] } }})
                    else:
                        logger.error("Unknown message type: %s", m)
            r = {
                'expectUserResponse': 'true',
                'expectedInputs': [{
                    'possibleIntents': [ { 'intent': 'actions.intent.TEXT' } ],
                    'inputPrompt': {
                        'richInitialPrompt': {
                            'items': items
                        }
                    }
                }]
            }
            return response.json(r)				
	
        return google_webhook

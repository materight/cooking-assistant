# Cooking Assistant with Rasa

## Get started
- Install the required dependencies and models:
    ```shell
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    ```
- Regenerate the NLU data (if modified): `python generate_data.py`
- Train the Rasa pipeline: `python -m rasa train`
- Start the Rasa actions server in a separate terminal: `python -m rasa run actions`
- Run the trained model: `python -m rasa shell`

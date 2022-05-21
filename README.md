# Cooking Assistant with Rasa

## Get started
- Install the required dependencies and models:
    ```shell
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    ```
- Regenerate the NLU data and train the Rasa pipeline:
    ```shell
    python -m rasa train
    python generate_data.py
    ```
- In two separate terminals, run the actions server and the trained model:
    ```shell
    python -m rasa run actions
    python -m rasa shell
    ```
## Hyperparameter optimization
- TODO

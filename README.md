# Cooking Assistant with Rasa
A *conversational agent* developed with [Rasa](https://rasa.com/) to help searching and preparing food recipes.

The NLU data is generated with [Chatette](https://github.com/SimGus/Chatette) and the recipes are provided by [justthedarnrecipe.com](https://justthedarnrecipe.com/).


## Get started
- Install the required dependencies and models:
    ```shell
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    ```
- Regenerate the NLU data and train the Rasa pipeline:
    ```shell
    python generate_data.py
    python -m rasa train
    ```
- In two separate terminals, run the actions server and the trained model:
    ```shell
    python -m rasa run actions
    python -m rasa shell
    ```


## Hyperparameter optimization
To run an hyperparameter search:

- Change the hyperparameters to use in `conig.hyperopt.yml`, under the `hyperparams` key.
- Run the `python hyperopt.py -n [N_ITERATIONS]` script.

For each configurations three runs will be executed, using different held-out fractions of the training data for evaluation. The configurations files, the trained models and the final evaluation results can be then found in the `hyperopts` directory.

After installing the necessary envriment, set your openai keys here:
src/alpaca_eval/constants.py

Or:
export OPENAI_API_KEY=<your_api_key>
export OPENAI_ORGANIZATION_IDS=<your_organization_id> 

Then directly run:
bash scripts_example/inference_eva.sh

Args:
model_name_tag: the name of the model, important, do not name different models with same names
model_name_or_path: the pathe of the model

When finish, go check to find the name of model:
src/alpaca_eval/leaderboards/data_AlpacaEval/alpaca_eval_gpt4_leaderboard.csv


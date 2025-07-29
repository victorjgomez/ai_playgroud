from transformers import pipeline
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

# Load a sentiment classification model
model_pipeline = pipeline("sentiment-analysis")
model = model_pipeline.model
tokenizer = model_pipeline.tokenizer

# Wrap the model for TextAttack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Define the attack (TextFooler is a common adversarial attack)
attack = TextFoolerJin2019.build(model_wrapper)

# Create a small dataset to test (label is just a placeholder)
dataset = Dataset([("I love this movie. It's amazing!", 1)])

# Configure the attack arguments
attack_args = AttackArgs(
    num_examples=1,
    log_to_csv="adversarial_results.csv",
    disable_stdout=True,
)

# Run the attack
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()

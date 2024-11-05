from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any


def load_mnli_train() -> Dataset:
    """you can use this for coming up with a few-shot prompt."""
    
    ds = load_dataset("facebook/anli", split="train_r3").take(100)
    return ds

def load_mnli_dev() -> Dataset:
    """Use this for picking your best verbalizer."""
    
    ds = load_dataset("facebook/anli", split="dev_r3").shuffle().take(50)
    return ds

def load_mnli_test() -> Dataset:
    """Use this only AFTER you have picked your best verbalizer."""
    
    ds = load_dataset("facebook/anli", split="test_r3").shuffle().take(100)
    return ds

def make_verbalizer(dev_ds: Dataset) -> str:
    zero_shot = """Given a premise and hypothesis, determine if the hypothesis is:
    - 0: hypothesis is implied by the premise
    - 1: hypothesis is neither contradicted nor implied by the premise
    - 2: hypothesis contradicts the premise

    only answer with 0, 1, or 2.
    """

    few_shot = """Given a premise and hypothesis, determine if the hypothesis is:
    - 0: hypothesis is implied by the premise
    - 1: hypothesis is neither contradicted nor implied by the premise
    - 2: hypothesis contradicts the premise

    Here are some examples:

    Premise: The Other One is the third solo album by former Fleetwood Mac guitarist Bob Welch. The track "Future Games" was first released on the Fleetwood Mac album of the same name in 1971.
    Hypothesis: Bob Welch is the current guitarist of Fleetwood Mac.
    Answer: 2

    Premise: TOKYO, Dec 18 (Reuters) - Japan’s Shionogi & Co said on Tuesday that it has applied to health regulators in the United States, Canada and Europe for approval of its HIV drug Dolutegravir. Shionogi developed Dolutegravir with a Viiv Healthcare, an AIDS drug joint venture between GlaxoSmithKline and Pfizer, in exchange for its rights to the drug.
    Hypothesis: The HIV drug is being approved for use in the usa
    Answer: 1

    Premise: Rhodochition is a genus of flowering plants within the family Plantaginaceae, native to southern Mexico and neighbouring Guatemala.
    Hypothesis: You can find the purple bell vine in more than one country.
    Answer: 0

    Now analyze this similar case, only answer with 0, 1, or 2:

    """
    verbalizer = zero_shot

    return verbalizer

def make_prompt(verbalizer: str, premise: str, hypothesis: str) -> str:
    
    prompt = f"{verbalizer}\nPremise: {premise}\nHypothesis: {hypothesis}\n"
    return prompt

def predict_labels(prompts: list[str]) -> list[int]:
    """Predicts labels for a list of prompts using a language model."""
    
    # Initialize the model and tokenizer
    model_name = "allenai/OLMo-7B-Instruct-hf"  # or another suitable model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    predictions = []
    for prompt in prompts:
        # Tokenize and generate completion
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            temperature=0.1,
            num_return_sequences=1,
        )
        
        # Extract the predicted label
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            # Extract the first number (0, 1, or 2) from the response
            label = int(next(char for char in response if char in "012"))
        except StopIteration:
            # Default to neutral if no valid label is found
            label = 1
            
        predictions.append(label)
    
    return predictions

if __name__ == "__main__":
    train_ds = load_mnli_train()
    dev_ds = load_mnli_dev()
    test_ds = load_mnli_test()
    
    verbalizer = make_verbalizer(train_ds)
    
    prompts = []
    true_labels = []
    for ex in dev_ds:
      prompt = make_prompt(verbalizer, ex["premise"], ex["hypothesis"])
      print(prompt)
      prompts.append(prompt)
      true_labels.append(ex["label"])
    
    predicted_labels = predict_labels(prompts)
    
    num_correct = sum(true == pred for true, pred in zip(true_labels, predicted_labels))
    accuracy = num_correct / len(true_labels)
    print("Accuracy:", accuracy)
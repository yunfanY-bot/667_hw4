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

    Premise: A 19-year-old Afghan asylum-seeker suffered serious burns after setting himself on fire at a supermarket warehouse in Bavaria, German police said. Police said that the man poured gasoline over himself and set himself ablaze early Monday in Gaimersheim, a town between Nuremberg and Munich. He had bought the gasoline shortly before at a filling station. The blaze was extinguished swiftly by other people at the scene, but the man was seriously injured. The man’s motives weren’t immediately clear. Police say he was carrying a knife but didn’t use it.
    Hypothesis: The injuries were fatal
    Answer: 2, hypothesis contradicts the premise

    Premise: A man walks past an electronic stock board showing Japan's Nikkei 225 index and other country's index at a securities firm in Tokyo Monday, Sept. 3, 2018. Asian shares were mostly lower Monday amid worries about escalating trade friction between the U.S. and Canada, who have been unable to agree to a revamped trade deal but will continue negotiating this week. Eugene Hoshiko AP Photo
    Hypothesis: The US Canada and Japan are close to concluding a trade deal.
    Answer: 1, hypothesis is neither contradicted nor implied by the premise

    Premise: by Ted Raymond, Newstalk 580 CFRA A stretch of Highway 17 between Pembroke and Mattawa has reopened, after being closed due to a fatal crash. Ontario Provincial Police say a motorcycle and a car collided just before 12:00 p.m. Friday, near Deux Rivieres. The motorcycle passenger was taken to hospital by air ambulance and was pronounced dead. The driver of the motorcycle was seriously injured. The driver of the other vehicle was not hurt. An OPP news release said westbound traffic was being diverted south on Highway 41 at Pembroke. The road reopened just after 6:30 p.m.
    Hypothesis: The road was closed for more than two hours after the crash
    Answer: 0, hypothesis is implied by the premise

    """
    verbalizer = few_shot

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
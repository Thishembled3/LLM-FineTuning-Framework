import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_seq_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.data = self._load_and_process_data(data_path)

    def _load_and_process_data(self, data_path):
        processed_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                # Assuming each example has 'text' and optionally 'label' or 'target_text'
                text = example.get('text', '')
                target_text = example.get('target_text', None)

                if target_text:
                    # For sequence-to-sequence tasks (e.g., summarization, translation)
                    model_inputs = self.tokenizer(text, max_length=self.max_seq_length, truncation=True)
                    labels = self.tokenizer(target_text, max_length=self.max_seq_length, truncation=True).input_ids
                    model_inputs["labels"] = labels
                else:
                    # For causal language modeling (e.g., text generation)
                    tokenized_input = self.tokenizer(text, max_length=self.max_seq_length, truncation=True)
                    model_inputs = {"input_ids": tokenized_input["input_ids"], "attention_mask": tokenized_input["attention_mask"]}
                    model_inputs["labels"] = tokenized_input["input_ids"].copy() # For causal LM, labels are usually input_ids

                processed_data.append(model_inputs)
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_tokenizer(self):
        return self.tokenizer

# Example usage:
if __name__ == "__main__":
    # Create a dummy JSONL file for testing
    dummy_data = [
        {"text": "The quick brown fox jumps over the lazy dog.", "target_text": "A fox jumps over a dog."},
        {"text": "Artificial intelligence is rapidly advancing.", "target_text": "AI is advancing."},
        {"text": "This is a test sentence for causal language modeling.", "target_text": None}
    ]
    os.makedirs("data", exist_ok=True)
    with open("data/custom_dataset.json", "w", encoding="utf-8") as f:
        for entry in dummy_data:
            f.write(json.dumps(entry) + "\n")

    # Test LLMDataset for sequence-to-sequence
    print("\nTesting LLMDataset for sequence-to-sequence task...")
    seq2seq_dataset = LLMDataset("data/custom_dataset.json", "t5-small")
    print(f"Dataset size: {len(seq2seq_dataset)}")
    sample = seq2seq_dataset[0]
    print(f"Sample input_ids: {sample["input_ids"][:10]}...")
    print(f"Sample labels: {sample["labels"][:10]}...")

    # Test LLMDataset for causal language modeling
    print("\nTesting LLMDataset for causal language modeling task...")
    causal_lm_dataset = LLMDataset("data/custom_dataset.json", "gpt2")
    print(f"Dataset size: {len(causal_lm_dataset)}")
    sample = causal_lm_dataset[2] # The third entry is for causal LM
    print(f"Sample input_ids: {sample["input_ids"][:10]}...")
    print(f"Sample labels: {sample["labels"][:10]}...")

    # Clean up dummy data
    os.remove("data/custom_dataset.json")
    os.rmdir("data")
// Update on 2023-06-22 00:00:00 - 711
// Update on 2023-10-20 00:00:00 - 233

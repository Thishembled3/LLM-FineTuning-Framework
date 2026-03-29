import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer as HfTrainer
from peft import LoraConfig, get_peft_model
from llm_finetuning.config import TrainingConfig
from llm_finetuning.data_processor import LLMDataset

class LLMFineTuner:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj"], # Common target modules for LLMs
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.train_dataset = LLMDataset(self.config.dataset_path, self.config.model_name, self.config.max_seq_length)

    def train(self):
        print(f"\n--- Starting LLM Fine-Tuning for {self.config.model_name} ---")
        self.config.display_config()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to="none", # Disable reporting to external services for this example
            remove_unused_columns=False, # Keep columns for data processing
        )

        trainer = HfTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._data_collator, # Custom data collator if needed
        )

        trainer.train()
        print("\n--- LLM Fine-Tuning Complete! ---")

    def _data_collator(self, features):
        # A simple data collator for demonstration. In a real scenario, you might need more complex padding.
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        return batch

# Example usage:
if __name__ == "__main__":
    # Ensure dummy data exists for testing
    import os
    import json
    os.makedirs("data", exist_ok=True)
    with open("data/custom_dataset.json", "w", encoding="utf-8") as f:
        dummy_data = [
            {"text": "This is a sample sentence for fine-tuning.", "target_text": "Sample output."},
            {"text": "Another example for the LLM.", "target_text": "Another output."},
            {"text": "A third sentence to test the model.", "target_text": "Third output."},
            {"text": "The quick brown fox jumps over the lazy dog.", "target_text": "A fox jumps over a dog."},
            {"text": "Artificial intelligence is rapidly advancing.", "target_text": "AI is advancing."},
            {"text": "This is a test sentence for causal language modeling.", "target_text": None}
        ]
        for entry in dummy_data:
            f.write(json.dumps(entry) + "\n")

    config = TrainingConfig()
    # For demonstration, let's set a smaller batch size and fewer epochs
    config.per_device_train_batch_size = 2
    config.num_train_epochs = 1
    config.model_name = "gpt2"

    finetuner = LLMFineTuner(config)
    finetuner.train()

    # Clean up dummy data
    os.remove("data/custom_dataset.json")
    os.rmdir("data")

import argparse
import os
from llm_finetuning.config import TrainingConfig
from llm_finetuning.trainer import LLMFineTuner
from llm_finetuning.data_processor import LLMDataset

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Framework")
    parser.add_argument("--model_name", type=str, default=None, help="Hugging Face model name or path")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for models and logs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=None, help="Dropout probability for LoRA layers")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length for tokenization")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every X updates steps")
    parser.add_argument("--logging_steps", type=int, default=None, help="Log every X updates steps")

    args = parser.parse_args()

    # Initialize configuration from environment variables or defaults
    config = TrainingConfig()

    # Override config with command-line arguments if provided
    if args.model_name: config.model_name = args.model_name
    if args.dataset_path: config.dataset_path = args.dataset_path
    if args.output_dir: config.output_dir = args.output_dir
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.num_train_epochs: config.num_train_epochs = args.num_train_epochs
    if args.per_device_train_batch_size: config.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_accumulation_steps: config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r: config.lora_r = args.lora_r
    if args.lora_alpha: config.lora_alpha = args.lora_alpha
    if args.lora_dropout: config.lora_dropout = args.lora_dropout
    if args.max_seq_length: config.max_seq_length = args.max_seq_length
    if args.save_steps: config.save_steps = args.save_steps
    if args.logging_steps: config.logging_steps = args.logging_steps

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize and run the fine-tuner
    finetuner = LLMFineTuner(config)
    finetuner.train()

if __name__ == "__main__":
    # Create a dummy JSONL file for testing if it doesn't exist
    os.makedirs("data", exist_ok=True)
    dummy_data_path = "data/custom_dataset.json"
    if not os.path.exists(dummy_data_path):
        import json
        dummy_data = [
            {"text": "This is a sample sentence for fine-tuning.", "target_text": "Sample output."},
            {"text": "Another example for the LLM.", "target_text": "Another output."},
            {"text": "A third sentence to test the model.", "target_text": "Third output."},
            {"text": "The quick brown fox jumps over the lazy dog.", "target_text": "A fox jumps over a dog."},
            {"text": "Artificial intelligence is rapidly advancing.", "target_text": "AI is advancing."},
            {"text": "This is a test sentence for causal language modeling.", "target_text": None}
        ]
        with open(dummy_data_path, "w", encoding="utf-8") as f:
            for entry in dummy_data:
                f.write(json.dumps(entry) + "\n")

    main()

    # Clean up dummy data after execution
    if os.path.exists(dummy_data_path):
        os.remove(dummy_data_path)
        os.rmdir("data")
// Update on 2024-02-28 00:00:00 - 73

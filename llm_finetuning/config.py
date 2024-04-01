import os

class TrainingConfig:
    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt2")
        self.dataset_path = os.getenv("LLM_DATASET_PATH", "data/custom_dataset.json")
        self.output_dir = os.getenv("LLM_OUTPUT_DIR", "./output")
        self.learning_rate = float(os.getenv("LLM_LEARNING_RATE", 2e-5))
        self.num_train_epochs = int(os.getenv("LLM_NUM_EPOCHS", 3))
        self.per_device_train_batch_size = int(os.getenv("LLM_TRAIN_BATCH_SIZE", 8))
        self.gradient_accumulation_steps = int(os.getenv("LLM_GRAD_ACCUM_STEPS", 1))
        self.lora_r = int(os.getenv("LLM_LORA_R", 8))
        self.lora_alpha = int(os.getenv("LLM_LORA_ALPHA", 16))
        self.lora_dropout = float(os.getenv("LLM_LORA_DROPOUT", 0.05))
        self.max_seq_length = int(os.getenv("LLM_MAX_SEQ_LENGTH", 512))
        self.save_steps = int(os.getenv("LLM_SAVE_STEPS", 500))
        self.logging_steps = int(os.getenv("LLM_LOGGING_STEPS", 100))

    def display_config(self):
        print("\n--- Training Configuration ---")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        print("----------------------------\n")

# Example usage:
if __name__ == "__main__":
    config = TrainingConfig()
    config.display_config()

    # You can override settings via environment variables
    os.environ["LLM_MODEL_NAME"] = "facebook/opt-125m"
    os.environ["LLM_LEARNING_RATE"] = "1e-4"
    updated_config = TrainingConfig()
    updated_config.display_config()
// Update on 2024-04-01 00:00:00 - 524

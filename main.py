from src.preprocess import transform
from src.fine_tune import fine_tune
from huggingface_hub import notebook_login

notebook_login()

def main():
    # The model that you want to train from the Hugging Face hub
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    new_model_name = "ai-digitizer"

    # Transforming raw data to training data
    print("Transforming raw data ...")
    data_set_name = "dataset_2024-08-22T20-01-33-524Z.json"

    raw_path = "data/raw/"
    transformed_path = "data/transformed/"

    dataset = transform(raw_path + data_set_name)
    dataset.save_to_disk(transformed_path + data_set_name)
    print("Raw data transformed!\n")

    # Spliting dataset into train and test data
    split_dataset = dataset['train'].train_test_split(test_size=0.2)  # 20% for test data
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # Fine-tuned model name
    print("Fine tuning ...")
    # Load the entire model on the GPU 0 (CUDA) or mps: apple silicon 
    device_map = "auto"

    fine_tune(base_model_name, train_dataset, test_dataset, new_model_name + "-adapter", device_map)
    print("New model created!")

    return

if __name__ == "__main__":
    main()
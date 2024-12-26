import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
from llshash import LSHash
from pprint import pprint


# Function to tokenize and pad the content
def tokenize_and_pad(content_list, tokenizer, max_length=2048):
    tokenized_output = tokenizer(
        content_list,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return tokenized_output["input_ids"]


# Initialize LSHash
ht = LSHash(
    hash_size=3,
    input_dim=2048,
    num_hashtables=14,
    matrices_filename="mat.npz",
    hashtable_filename="hash.npz",
    num_cpus=4,
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("amirakhlaghiqqq/persian-llama2")


def mapper(batch):
    ted_and_ped = tokenize_and_pad(batch["text"], tokenizer)
    ht.index_batch(ted_and_ped, np.array(batch["index"]))
    ht.save_batch()
    ht.restart()
    return batch


def list_to_hf_dataset_with_index(list_of_strings):
    # Create a dictionary with the list of strings and an index column
    data_dict = {
        "index": list(range(1, len(list_of_strings) + 1)),
        "text": list_of_strings,
    }

    # Create a Hugging Face dataset from the dictionary
    dataset = Dataset.from_dict(data_dict)

    return dataset


# Example usage
if __name__ == "__main__":
    list_of_strings = [
        # Your list of Persian strings here...
    ]
    data = list_to_hf_dataset_with_index(list_of_strings)

    mapped_data = data.map(mapper, batched=True, batch_size=7)
    ht.load_batch()
    pairs = ht.find_similar_documents()
    print(pairs)
    pprint(ht.hash_tables)

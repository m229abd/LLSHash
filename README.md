# LLSHash-Py

**Version:** 1.0.0

LLSHash-Py is an enhanced Python implementation of Locality Sensitive Hashing (LSH) tailored for data deduplication in large-scale datasets, particularly for training large language models. Developed at Shahid Beheshti University, this package leverages modern industry-standard libraries and parallel processing techniques to ensure efficient and scalable performance.

## Highlights

- **Direct Duplicate Finder**: Quickly identify and manage duplicate data points.
- **Batch Processing**: Improved performance through efficient batch indexing.
- **Parallel Processing**: Utilize CPUs and clusters with Ray for scalable hashing.
- **Integration with Hugging Face Suite**: Seamlessly integrate with Hugging Face libraries for advanced data handling.
- **Disk-Based Storage Management**: Offload and reload data to disk to manage RAM usage in gigantic datasets.
- **Unique Identifier Tracking**: Track indexed points with unique identifiers, simplifying data point management in large datasets.
- **NumPy's Advanced Storage**: Leverage NumPy's latest storage management features.
- **Removed Redis Dependency**: Streamlined storage management without Redis.
- **Modern Best Practices**: Incorporates type annotations and removes support for deprecated packages.

## Installation

LLSHash-Py depends on the following libraries:

- `numpy`
- `ray`
- `transformers`
- `datasets`

To install the package via pip:

```bash
pip install llshash-py
```

**Note:** Ensure that Ray is properly set up for parallel processing, especially if you intend to use cluster features.

## Quickstart

To create 6-bit hashes for input data of 8 dimensions and perform batch indexing:

```python
from llshash import LLSHash
import numpy as np

# Initialize LLSHash
lsh = LLSHash(
    hash_size=6,
    input_dim=8,
    num_hashtables=2,
    matrices_filename='matrices.npz',
    hashtable_filename='hashtables.npz',
    overwrite=True,
    num_cpus=4
)

# Index a single data point
input_point = np.random.rand(8)
extra_data = 1
lsh.index(input_point, extra_data)

# Batch indexing
input_points = np.random.rand(1000, 8)
extra_data_batch = np.arange(1000)
lsh.index_batch(input_points, extra_data_batch)

# Find similar documents
similar_pairs = lsh.find_similar_documents()
print(similar_pairs)

# Save hash tables
lsh.save()

# Load hash tables
lsh.load_batch()
```

## Detailed Usage

### Initializing LLSHash

```python
from llshash import LLSHash

lsh = LLSHash(
    hash_size=6,
    input_dim=8,
    num_hashtables=2,
    matrices_filename='matrices.npz',
    hashtable_filename='hashtables.npz',
    overwrite=True,
    num_cpus=4
)
```

**Parameters:**

- `hash_size`: Length of the resulting binary hash (e.g., 6 bits).
- `input_dim`: Dimension of the input vector (e.g., 8).
- `num_hashtables`: Number of hash tables used for multiple lookups (default: 1).
- `matrices_filename`: Path to the `.npz` file where random matrices are stored.
- `hashtable_filename`: Path to the `.npz` file where hash tables are stored.
- `overwrite`: Whether to overwrite existing matrix files (default: False).
- `num_cpus`: Number of CPUs to use for parallel processing (default: 1).

### Indexing Data Points

#### Single Indexing

```python
import numpy as np

input_point = np.random.rand(8)
extra_data = 1
lsh.index(input_point, extra_data)
```

#### Batch Indexing

```python
input_points = np.random.rand(1000, 8)
extra_data_batch = np.arange(1000)
lsh.index_batch(input_points, extra_data_batch)
```

### Finding Similar Documents

```python
similar_pairs = lsh.find_similar_documents()
print(similar_pairs)
```

### Saving and Loading Hash Tables

```python
# Save hash tables
lsh.save()

# Load hash tables
lsh.load_batch()
```

## Example with Hugging Face Libraries

Integration with Hugging Face's Transformers and Datasets for advanced data handling.

```python
from transformers import AutoTokenizer
from datasets import Dataset
from llshash import LLSHash
import numpy as np
from pprint import pprint

# Function to tokenize and pad the content
def tokenize_and_pad(content_list, tokenizer, max_length=2048):
    tokenized_output = tokenizer(content_list, padding='max_length', truncation=True, max_length=max_length, return_tensors='np')
    return tokenized_output['input_ids']

# Initialize LLSHash
ht = LLSHash(
    hash_size=6,
    input_dim=2048,
    num_hashtables=14,
    matrices_filename='mat.npz',
    hashtable_filename='hash.npz',
    num_cpus=4
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('amirakhlaghiqqq/persian-llama2')

def mapper(batch):
    ted_and_ped = tokenize_and_pad(batch["text"], tokenizer)
    ht.index_batch(ted_and_ped, np.array(batch['index']))
    ht.save_batch()
    ht.restart()
    return batch

def list_to_hf_dataset_with_index(list_of_strings):
    data_dict = {
        "index": list(range(1, len(list_of_strings) + 1)),
        "text": list_of_strings
    }
    dataset = Dataset.from_dict(data_dict)
    return dataset

# Example usage
if __name__ == "__main__":
    list_of_strings = [
        # Your test set for manual incpetion
    ]
    data = list_to_hf_dataset_with_index(list_of_strings)
    mapped_data = data.map(mapper, batched=True, batch_size=7)
    ht.load_batch()
    pairs = ht.find_similar_documents()
    print(pairs)
    pprint(ht.hash_tables)
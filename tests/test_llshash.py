import numpy as np
import pytest
import ray


from llshash import LLSHash


@pytest.fixture(scope="module")
def ray_init():
    """Ensure Ray is initialized before tests and properly shut down after."""
    ray.init()
    yield
    ray.shutdown()


@pytest.fixture
def tmp_files(tmp_path):
    """
    Create temporary file paths for the .npz files.
    Pytest will ensure they are cleaned up after tests.
    """
    matrices_file = str(tmp_path / "mat_test.npz")
    hashtable_file = str(tmp_path / "hash_test.npz")
    return matrices_file, hashtable_file


@pytest.fixture
def example_lsh(tmp_files):
    """Initialize LLSHash instance with small parameters for quick testing."""
    matrices_file, hashtable_file = tmp_files
    lsh = LLSHash(
        hash_size=3,
        input_dim=10,  # small dimension for faster test
        num_hashtables=2,
        matrices_filename=matrices_file,
        hashtable_filename=hashtable_file,
        overwrite=True,
        num_cpus=2,
    )
    return lsh


def test_init_lsh(example_lsh):
    """Test that LLSHash initializes correctly."""
    lsh = example_lsh
    assert lsh.hash_size == 3
    assert lsh.input_dim == 10
    assert (
        len(lsh.uniform_planes) == 2
    ), "Should have 2 sets of uniform planes."
    assert len(lsh.hash_tables) == 2, "Should have 2 hash tables."


def test_planes_saved_and_loaded(tmp_files):
    """
    Test that uniform planes are saved to an .npz file and loaded correctly if
    not overwritten.
    """
    matrices_file, hashtable_file = tmp_files

    # 1) Create an LSH and allow it to save the planes.
    lsh1 = LLSHash(
        3,
        10,
        2,
        matrices_filename=matrices_file,
        hashtable_filename=hashtable_file,
        overwrite=True,
    )
    # 2) Generate planes.
    planes_before = [plane.copy() for plane in lsh1.uniform_planes]

    # 3) Create another instance with overwrite=False (load existing planes).
    lsh2 = LLSHash(
        3,
        10,
        2,
        matrices_filename=matrices_file,
        hashtable_filename=hashtable_file,
        overwrite=False,
    )

    for p1, p2 in zip(planes_before, lsh2.uniform_planes):
        np.testing.assert_array_almost_equal(
            p1,
            p2,
            decimal=6,
            err_msg="Loaded planes must match previously saved planes",
        )


def test_hash_tables_saved_and_loaded(example_lsh, tmp_path):
    """
    Test that hash tables can be saved and loaded properly.
    """
    lsh = example_lsh

    # Index some data
    point = np.random.rand(1, lsh.input_dim)  # single point
    lsh.index(point[0], extra_data=1001)

    # Save the hashtable
    lsh.save()

    # Clear in-memory
    for table in lsh.hash_tables:
        table.clear()
    assert all(
        [len(table) == 0 for table in lsh.hash_tables]
    ), "Hash tables should be empty after clear."

    # Load from file again
    lsh._init_hashtables()

    # The loaded data should still have the entry we indexed
    found = False
    for table in lsh.hash_tables:
        for key, arr in table.items():
            if 1001 in arr:
                found = True
                break
    assert (
        found
    ), "Saved/loaded hash tables should contain previously indexed data."


def test_hash_function(example_lsh):
    """
    Test that the _hash function produces a binary string of length hash_size.
    """
    lsh = example_lsh
    test_point = np.random.rand(lsh.input_dim)
    generated_hash = lsh._hash(lsh.uniform_planes[0], test_point)
    assert (
        len(generated_hash) == lsh.hash_size
    ), "Hash length must match hash_size."
    assert set(generated_hash).issubset(
        {"0", "1"}
    ), "Hash must be binary (0 or 1)."


def test_index_single_point(example_lsh):
    """
    Test indexing of a single point.
    Ensure it’s placed in the right bin in the hash table.
    """
    lsh = example_lsh
    test_point = np.ones(lsh.input_dim) * 0.5
    result_hashes = lsh.index(test_point, extra_data=42)

    assert (
        len(result_hashes) == lsh.num_hashtables
    ), "Should return one hash per table."

    # Check if 42 is in each corresponding bucket
    for i, table in enumerate(lsh.hash_tables):
        h = result_hashes[i]
        assert h in table, f"Hash {h} must be in table {i}."
        assert 42 in table[h], f"extra_data=42 must appear in the bin {h}."


def test_batch_index(ray_init, example_lsh):
    """
    Test indexing a batch of points
    This also tests the Ray remote function _hash_batch.
    """
    lsh = example_lsh
    # Create a batch of 5 points
    batch_points = np.random.randn(5, lsh.input_dim)
    extra_data = np.array([10, 20, 30, 40, 50])

    lsh.index_batch(batch_points, extra_data)

    # Verify that each extra_data item is found in the hash tables
    found_items = []
    for table in lsh.hash_tables:
        for arr in table.values():
            found_items.extend(arr.tolist())
    found_items = set(found_items)

    for item in extra_data:
        assert (
            item in found_items
        ), f"Item {item} not found in the hash tables after batch index."


def test_save_and_load_batch(ray_init, example_lsh, tmp_path):
    """
    Test the batch saving logic:
    1) index some data
    2) save batch
    3) restart
    4) load batch
    5) ensure data is combined from all saved files
    """
    lsh = example_lsh
    # Index a first batch
    batch1 = np.random.randn(3, lsh.input_dim)
    extra_data1 = np.array([1, 2, 3])
    lsh.index_batch(batch1, extra_data1)
    lsh.save_batch()

    # Index a second batch
    batch2 = np.random.randn(2, lsh.input_dim)
    extra_data2 = np.array([4, 5])
    lsh.index_batch(batch2, extra_data2)
    lsh.save_batch()

    # Clear and restart in memory
    lsh.restart()
    assert all(
        [len(table) == 0 for table in lsh.hash_tables]
    ), "Hash tables should be empty after restart."

    # Load all batches
    lsh.load_batch()

    # All items from 1..5 should appear
    found_items = []
    for table in lsh.hash_tables:
        for arr in table.values():
            found_items.extend(arr)
    found_items = set(found_items)

    for item in [1, 2, 3, 4, 5]:
        assert (
            item in found_items
        ), f"Item {item} was not found after loading all batch files."


def test_find_similar_documents(ray_init, example_lsh):
    """
    Test the logic for find_similar_documents.
    We will index data so that some points are always in the same bucket.
    Then see if the method returns them as a group.
    """
    lsh = example_lsh

    # Clear any existing data
    for table in lsh.hash_tables:
        table.clear()

    # Let’s create 4 points:
    # - points 0 & 1 are identical => they should end up in the same bins
    # - points 2 & 3 are random => less likely to appear in the same bins
    data_points = np.array(
        [
            np.ones(lsh.input_dim) * 0.1,  # point 0
            np.ones(lsh.input_dim) * 0.1,  # point 1 (identical to 0)
            np.random.rand(lsh.input_dim),  # point 2
            np.random.rand(lsh.input_dim),  # point 3
        ]
    )
    extra_data = np.array([100, 101, 102, 103])

    lsh.index_batch(data_points, extra_data)

    # find_similar_documents should group 100 & 101 together
    # 102 & 103 might not get grouped (unless randomly placed together in all
    # hashtables).
    similar_groups = lsh.find_similar_documents()

    # We expect at least one group that has [100, 101]
    found_identical = any(
        [set(group) == {100, 101} for group in similar_groups]
    )
    assert found_identical, "Points 100 & 101 should appear in the same group."

    # It's possible 102 & 103 won't appear or appear singly.
    # The test only checks if 100 & 101 definitely appear together.

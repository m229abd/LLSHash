import os
import numpy as np
import glob
from typing import List, Optional, Dict, Set
import ray


@ray.remote
def _hash_batch(
    planes: np.ndarray,
    input_points: np.ndarray,
    table: Dict[str, np.ndarray],
    extra_data: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Generate binary hashes for input points and update the hash table.

    Projects input points onto hyperplanes, creates binary hashes, and appends
    extra data to the table.

    Parameters
    ----------
    planes : np.ndarray
        Hyperplanes for hashing, shape (hash_size, input_dim).
    input_points : np.ndarray
        Points to hash, shape (input_dim, num_points).
    table : Dict[str, np.ndarray]
        Current hash table mapping hashes to data indices.
    extra_data : np.ndarray
        Data indices to associate with each input point.

    Returns
    -------
    Dict[str, np.ndarray]
        Updated hash table with new entries.
    """
    projections = np.dot(planes, input_points.T)
    binary_hashes = (projections > 0).astype(int)
    for j, h in enumerate(binary_hashes.T):
        h_str = "".join(map(str, h))
        if h_str not in table:
            table[h_str] = np.array([], dtype=int)
        table[h_str] = np.append(table[h_str], extra_data[j])
    return table


class LLSHash:
    """
    Locality Sensitive Hashing using random hyperplanes.

    Facilitates efficient indexing and similarity search of high-dimensional
    data.
    """

    def __init__(
        self,
        hash_size: int,
        input_dim: int,
        num_hashtables: int = 1,
        matrices_filename: Optional[str] = None,
        hashtable_filename: Optional[str] = None,
        overwrite: bool = False,
        num_cpus: int = 1,
    ) -> None:
        """
        Initialize LLSHash with hashing parameters and optional file paths.

        Parameters
        ----------
        hash_size : int
            Number of hyperplanes per hash table.
        input_dim : int
            Dimensionality of input vectors.
        num_hashtables : int, optional
            Number of hash tables to use, default is 1.
        matrices_filename : Optional[str], optional
            Path to load/save hyperplane matrices, default is None.
        hashtable_filename : Optional[str], optional
            Path to load/save hash tables, default is None.
        overwrite : bool, optional
            Whether to overwrite existing matrices, default is False.
        num_cpus : int, optional
            Number of CPU cores for parallel operations, default is 1.
        """
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.counter = 0
        self.matrices_filename = matrices_filename
        self.hashtable_filename = hashtable_filename
        self.overwrite = overwrite
        self.num_cpus = num_cpus

        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self) -> None:
        """
        Initialize or load hyperplanes for hashing.
        """
        if hasattr(self, "uniform_planes"):
            return

        if (
            self.matrices_filename
            and os.path.isfile(self.matrices_filename)
            and not self.overwrite
        ):
            try:
                npzfiles = np.load(self.matrices_filename)
                self.uniform_planes = sorted(
                    npzfiles.values(), key=lambda x: x.shape[0]
                )
            except IOError:
                raise IOError("Failed to load hyperplane matrices.")
        else:
            self.uniform_planes = [
                self._generate_uniform_planes()
                for _ in range(self.num_hashtables)
            ]
            if self.matrices_filename:
                try:
                    np.savez_compressed(
                        self.matrices_filename, *self.uniform_planes
                    )
                except IOError:
                    raise IOError("Failed to save hyperplane matrices.")

    def _init_hashtables(self) -> None:
        """
        Initialize or load hash tables.
        """
        if self.hashtable_filename and os.path.isfile(self.hashtable_filename):
            try:
                npzfiles = np.load(self.hashtable_filename, allow_pickle=True)
                self.hash_tables = npzfiles["data"]
            except IOError:
                raise IOError("Failed to load hash tables.")
        else:
            self.hash_tables = [{} for _ in range(self.num_hashtables)]

    def _generate_uniform_planes(self) -> np.ndarray:
        """
        Create random hyperplanes for hashing.

        Returns
        -------
        np.ndarray
            Random hyperplanes, shape (hash_size, input_dim).
        """
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes: np.ndarray, input_point: np.ndarray) -> str:
        """
        Compute binary hash for a single input point.

        Parameters
        ----------
        planes : np.ndarray
            Hyperplanes for hashing.
        input_point : np.ndarray
            The input vector to hash.

        Returns
        -------
        str
            Binary hash string.

        Raises
        ------
        TypeError
            If input_point contains non-numeric data.
        ValueError
            If input_point dimensionality doesn't match input_dim.
        """
        try:
            projections = np.dot(planes, input_point)
        except TypeError:
            raise TypeError("Input point must contain numeric values.")
        except ValueError:
            raise ValueError("Input point dimensionality mismatch.")
        return "".join(["1" if p > 0 else "0" for p in projections])

    def index(self, input_point: np.ndarray, extra_data: int) -> List[str]:
        """
        Add a single input point to all hash tables.

        Parameters
        ----------
        input_point : np.ndarray
            Vector to index.
        extra_data : int
            Identifier associated with the input point.

        Returns
        -------
        List[str]
            List of hash strings for each table.
        """
        hashes = []
        for i, table in enumerate(self.hash_tables):
            h = self._hash(self.uniform_planes[i], input_point)
            table.setdefault(h, np.array([], dtype=int))
            table[h] = np.append(table[h], extra_data)
            hashes.append(h)
        return hashes

    def index_batch(
        self, input_points: np.ndarray, extra_data: np.ndarray
    ) -> None:
        """
        Index multiple input points using parallel processing.

        Parameters
        ----------
        input_points : np.ndarray
            Vectors to index, shape (input_dim, num_points).
        extra_data : np.ndarray
            Identifiers for each input point.

        Raises
        ------
        ValueError
            If number of input points doesn't match extra_data length.
        """
        if input_points.shape[1] != extra_data.shape[0]:
            raise ValueError("Mismatch between points and extra_data lengths.")

        ray.init(ignore_reinit_error=True)
        futures = [
            _hash_batch.options(
                num_cpus=self.num_cpus
            ).remote(  # type: ignore[attr-defined]
                self.uniform_planes[i], input_points, table, extra_data
            )
            for i, table in enumerate(self.hash_tables)
        ]
        self.hash_tables = ray.get(futures)

    def save(self, batch: bool = False) -> None:
        """
        Save hash tables to file.

        Parameters
        ----------
        batch : bool, optional
            If True, appends the counter to the filename, default is False.

        Raises
        ------
        IOError
            If saving fails.
        ValueError
            If filename is not specified when batch is True.
        """
        filename: Optional[str] = None  # Initialize once

        if batch:
            if not self.hashtable_filename:
                raise ValueError("Filename required for batch saving.")
            base, ext = os.path.splitext(self.hashtable_filename)
            filename = f"{base}_{self.counter}{ext}"
        else:
            filename = self.hashtable_filename

        if filename:
            try:
                np.savez_compressed(filename, data=self.hash_tables)
            except IOError:
                raise IOError("Failed to save hash tables.")
        else:
            raise ValueError("No filename specified for saving.")

    def save_batch(self) -> None:
        """
        Save current hash tables as a batch and increment the counter.
        """
        self.save(batch=True)
        self.counter += 1

    def load_batch(self) -> None:
        """
        Load all batch files and merge them into the current hash tables.

        Raises
        ------
        IOError
            If loading any batch file fails.
        ValueError
            If hashtable_filename is not specified.
        """
        if not self.hashtable_filename:
            raise ValueError(
                "Hashtable filename must be specified to load batches."
            )

        base, ext = os.path.splitext(self.hashtable_filename)
        pattern = f"{base}_*.npz"
        files = glob.glob(pattern)
        self.hash_tables = [{} for _ in range(self.num_hashtables)]
        for file in files:
            try:
                npzfiles = np.load(file, allow_pickle=True)
                for i in range(self.num_hashtables):
                    for key, values in npzfiles["data"][i].items():
                        self.hash_tables[i].setdefault(
                            key, np.array([], dtype=int)
                        )
                        self.hash_tables[i][key] = np.append(
                            self.hash_tables[i][key], values
                        )
            except IOError:
                raise IOError(f"Failed to load hash table from {file}.")

    def restart(self) -> None:
        """
        Clear all hash tables and reset the counter.
        """
        for table in self.hash_tables:
            table.clear()
        self.counter += 1

    def get_hashes(self, input_point: np.ndarray) -> List[str]:
        """
        Retrieve hash strings for an input point across all tables.

        Parameters
        ----------
        input_point : np.ndarray
            The input vector.

        Returns
        -------
        List[str]
            List of hash strings.
        """
        return [
            self._hash(planes, input_point) for planes in self.uniform_planes
        ]

    def find_similar_documents(self) -> List[List[int]]:
        """
        Find groups of similar documents based on hash collisions.

        Returns
        -------
        List[List[int]]
            Groups of similar document indices.
        """
        if self.num_hashtables <= 1:
            print("At least two hash tables are required.")
            return []

        candidate_pairs: Dict[int, Set[int]] = {}
        for table in self.hash_tables:
            for indices in table.values():
                if len(indices) > 1:
                    for idx in indices:
                        candidate_pairs.setdefault(idx, set()).update(indices)
        similar_documents = []
        checked = set()
        for idx, candidates in candidate_pairs.items():
            for cand in candidates:
                if idx != cand and (idx, cand) not in checked:
                    if all(
                        any(
                            idx in vals and cand in vals
                            for vals in table.values()
                        )
                        for table in self.hash_tables
                    ):
                        similar_documents.append([idx, cand])
                        checked.add((idx, cand))
                        checked.add((cand, idx))
        merged: List[Set[int]] = []
        for group in similar_documents:
            merged_found = False
            for m in merged:
                if set(group) & set(m):
                    m.update(group)
                    merged_found = True
                    break
            if not merged_found:
                merged.append(set(group))

        return [sorted(list(group)) for group in merged]

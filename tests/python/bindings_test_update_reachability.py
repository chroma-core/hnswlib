import unittest

import numpy as np

import hnswlib


class UpdateReachabilityTestCase(unittest.TestCase):
    def testUpdatesKeepElementsReachable(self):
        """
        Tests that updating vectors of existing elements, with several threads, leaves every element reachable:
        a query of all elements returns all of them, and each element is its own nearest neighbor.
        """
        num_elements = 60
        num_updated = 20
        dim = 4
        trials = 1000

        rng = np.random.default_rng(0)
        unreachable = 0
        for _ in range(trials):
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=100, M=16)
            index.set_ef(100)
            data = np.float32(rng.uniform(-1, 1, (num_elements, dim)))
            labels = np.arange(num_elements)
            index.add_items(data, labels, num_threads=4)

            updated = rng.choice(num_elements, num_updated, replace=False)
            data[updated] = np.float32(rng.uniform(-1, 1, (num_updated, dim)))
            index.add_items(data[updated], labels[updated], num_threads=4)

            query = np.float32(rng.uniform(-1, 1, (1, dim)))
            try:
                found, _ = index.knn_query(query, k=num_elements)
                complete = len(set(found[0])) == num_elements
            except RuntimeError:
                # The query found fewer than k elements: some element is not reachable.
                complete = False
            own, _ = index.knn_query(data, k=1)
            if not complete or not np.array_equal(own[:, 0], labels):
                unreachable += 1

        self.assertEqual(unreachable, 0, f"{unreachable} of {trials} trials left an element unreachable")


if __name__ == "__main__":
    unittest.main()

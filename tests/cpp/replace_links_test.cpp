#include "../../hnswlib/hnswlib.h"

#include <iostream>
#include <vector>

// Replacing a deleted element must not give the new element the links of the old one: the new element's list on
// layer 0 holds only elements near its own vector.
int testReplacedElementLinksNearItsVector()
{
    const int dim = 2;
    const int clusterSize = 20;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 2 * clusterSize + 1, 4, 50, 100, true);

    // Cluster A around (0, 0), cluster B around (100, 100).
    std::vector<float> point(dim);
    for (int i = 0; i < 2 * clusterSize; i++)
    {
        float base = i < clusterSize ? 0.0f : 100.0f;
        point[0] = base + (i % 5);
        point[1] = base + (i % clusterSize) / 5;
        index.addPoint(point.data(), i);
    }

    // The element of label 0, in cluster A, is deleted and its slot reused by an element in cluster B.
    hnswlib::tableint replaced = index.label_lookup_[0];
    index.markDelete(0);
    point[0] = 100.5f;
    point[1] = 100.5f;
    index.addPoint(point.data(), 1000, true);
    hnswlib::tableint reused = index.label_lookup_[1000];
    if (reused != replaced)
    {
        std::cout << "The new element did not reuse the slot of the deleted element\n";
        return 1;
    }

    hnswlib::linklistsizeint *list = index.get_linklist0(reused);
    int size = index.getListCount(list);
    hnswlib::tableint *links = (hnswlib::tableint *)(list + 1);
    for (int i = 0; i < size; i++)
    {
        float *neighbor = (float *)index.getDataByInternalId(links[i]);
        if (neighbor[0] < 50.0f)
        {
            std::cout << "The reused element links to (" << neighbor[0] << ", " << neighbor[1]
                      << "), a neighbor of the element it replaced\n";
            return 1;
        }
    }
    std::cout << "Replaced element has " << size << " links, all near its own vector\n";
    return 0;
}

// When every other element is deleted, the search finds no element to link to: the reused slot must not keep the
// links of the element it replaced either.
int testReplacedElementWithEveryOtherDeleted()
{
    const int dim = 2;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 2, 16, 200, 100, true);

    std::vector<float> point = {0.0f, 0.0f};
    index.addPoint(point.data(), 0);
    point = {1.0f, 1.0f};
    index.addPoint(point.data(), 1);
    index.markDelete(0);
    index.markDelete(1);

    point = {5.0f, 5.0f};
    index.addPoint(point.data(), 2, true);
    hnswlib::tableint reused = index.label_lookup_[2];
    for (int level = 0; level <= index.element_levels_[reused]; level++)
    {
        int size = index.getListCount(index.get_linklist_at_level(reused, level));
        if (size != 0)
        {
            std::cout << "The reused element keeps " << size << " links of the element it replaced on layer " << level
                      << "\n";
            return 1;
        }
    }
    std::cout << "Replaced element with every other element deleted has no links\n";
    return 0;
}

int main()
{
    int failed = testReplacedElementLinksNearItsVector();
    failed += testReplacedElementWithEveryOtherDeleted();
    return failed == 0 ? 0 : 1;
}

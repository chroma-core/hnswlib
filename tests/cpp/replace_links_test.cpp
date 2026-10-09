#include "../../hnswlib/hnswlib.h"

#include <assert.h>
#include <iostream>
#include <vector>

// Replacing a deleted element must not give the new element the links of the old one: the new element's list on
// layer 0 holds only elements near its own vector.
int main()
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
    assert(reused == replaced);

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

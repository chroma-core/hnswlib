#include "../../hnswlib/hnswlib.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Updating the vector of an element must not drop a link from a list that has room for it: every list keeps the
// links it held before the update. With fewer elements than a list can hold, no list is ever full, so no update may
// drop a link.
int main()
{
    const int dim = 4;
    const int numElements = 30;
    const int numUpdates = 100;
    hnswlib::L2Space space(dim);
    // M = 16: a list holds up to 32 links on layer 0 and 16 above, more than the elements of each layer.
    hnswlib::HierarchicalNSW<float> index(&space, numElements, 16, 100);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> point(dim);
    for (int i = 0; i < numElements; i++)
    {
        for (float &x : point)
            x = uniform(rng);
        index.addPoint(point.data(), i);
    }

    for (int update = 0; update < numUpdates; update++)
    {
        std::vector<std::vector<std::vector<hnswlib::tableint>>> before(numElements);
        for (hnswlib::tableint id = 0; id < numElements; id++)
        {
            for (int level = 0; level <= index.element_levels_[id]; level++)
                before[id].push_back(index.getConnectionsWithLock(id, level));
        }

        hnswlib::labeltype label = update % numElements;
        for (float &x : point)
            x = uniform(rng);
        index.addPoint(point.data(), label);

        for (hnswlib::tableint id = 0; id < numElements; id++)
        {
            for (int level = 0; level <= index.element_levels_[id]; level++)
            {
                std::vector<hnswlib::tableint> after = index.getConnectionsWithLock(id, level);
                for (hnswlib::tableint link : before[id][level])
                {
                    if (std::find(after.begin(), after.end(), link) == after.end())
                    {
                        std::cout << "Updating label " << label << " dropped the link from " << id << " to " << link
                                  << " on layer " << level << ", in a list of " << after.size() << " links\n";
                        return 1;
                    }
                }
            }
        }
    }
    std::cout << numUpdates << " updates kept every link\n";
    return 0;
}

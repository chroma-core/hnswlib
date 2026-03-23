#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    std::vector<int> inbound_counts(const hnswlib::HierarchicalNSW<float> &index)
    {
        std::vector<int> inbound(index.cur_element_count, 0);
        for (size_t i = 0; i < index.cur_element_count; ++i)
        {
            for (int level = 0; level <= index.element_levels_[i]; ++level)
            {
                hnswlib::linklistsizeint *link_list = index.get_linklist_at_level(i, level);
                int size = index.getListCount(link_list);
                hnswlib::tableint *neighbors = reinterpret_cast<hnswlib::tableint *>(link_list + 1);
                for (int j = 0; j < size; ++j)
                {
                    inbound[neighbors[j]]++;
                }
            }
        }
        return inbound;
    }

    void remove_inbound_references(hnswlib::HierarchicalNSW<float> &index, hnswlib::tableint target)
    {
        for (size_t i = 0; i < index.cur_element_count; ++i)
        {
            for (int level = 0; level <= index.element_levels_[i]; ++level)
            {
                hnswlib::linklistsizeint *link_list = index.get_linklist_at_level(i, level);
                int size = index.getListCount(link_list);
                hnswlib::tableint *neighbors = reinterpret_cast<hnswlib::tableint *>(link_list + 1);

                int write = 0;
                for (int read = 0; read < size; ++read)
                {
                    if (neighbors[read] != target)
                    {
                        neighbors[write++] = neighbors[read];
                    }
                }
                index.setListCount(link_list, write);
            }
        }
    }

    void testCheckIntegritySkipsDeletedNodesInInboundStats()
    {
        const int d = 8;
        const int n = 32;
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
        std::vector<float> data(n * d);
        for (float &value : data)
        {
            value = distrib(rng);
        }

        hnswlib::L2Space space(d);
        hnswlib::HierarchicalNSW<float> index(&space, n, 8, 40, 17);
        for (int i = 0; i < n; ++i)
        {
            index.addPoint(data.data() + i * d, i);
        }

        std::vector<int> before = inbound_counts(index);
        for (int count : before)
        {
            assert(count > 0);
        }

        index.markDelete(0);
        remove_inbound_references(index, 0);

        std::vector<int> after = inbound_counts(index);
        assert(after[0] == 0);

        int expected_min = after[1];
        int expected_max = after[1];
        for (int i = 1; i < n; ++i)
        {
            assert(after[i] > 0);
            expected_min = std::min(expected_min, after[i]);
            expected_max = std::max(expected_max, after[i]);
        }

        std::ostringstream captured;
        std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
        index.checkIntegrity();
        std::cout.rdbuf(old);

        const std::string output = captured.str();
        const std::string expected_line =
            "Min inbound: " + std::to_string(expected_min) + ", Max inbound:" + std::to_string(expected_max);
        assert(output.find(expected_line) != std::string::npos);
        assert(output.find("Min inbound: 0") == std::string::npos);
    }
} // namespace

int main()
{
    std::cout << "Testing ..." << std::endl;
    testCheckIntegritySkipsDeletedNodesInInboundStats();
    std::cout << "Test testCheckIntegritySkipsDeletedNodesInInboundStats ok" << std::endl;
    return 0;
}

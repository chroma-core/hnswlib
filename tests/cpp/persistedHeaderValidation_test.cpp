#include "../../hnswlib/hnswlib.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>

namespace
{
    // size_data_per_element_ is the fifth header field, after the version and
    // three size_t fields (offsetLevel0_, max_elements_, cur_element_count).
    const std::streamoff kSizeDataPerElementOffset = sizeof(int) + 3 * sizeof(size_t);
    // maxM0_ is the eleventh header field: version, seven size_t, int, tableint,
    // then maxM_ before it.
    const std::streamoff kMaxM0Offset = sizeof(int) + 7 * sizeof(size_t) + sizeof(int) + sizeof(hnswlib::tableint);
    const int kDim = 8;
    const size_t kCount = 16;

    int failures = 0;

    void check(bool condition, const std::string &what)
    {
        // Not assert(): the release flags define NDEBUG, which would compile
        // every assertion out and leave this test passing unconditionally.
        if (condition)
        {
            std::cout << "    ok: " << what << std::endl;
            return;
        }
        std::cout << "    FAILED: " << what << std::endl;
        failures++;
    }

    void buildPersistedIndex(const std::string &dir)
    {
        hnswlib::InnerProductSpace space(kDim);
        hnswlib::HierarchicalNSW<float> index(&space, 2 * kCount, 16, 200, 100, false, false, true, dir);
        std::vector<float> v(kDim, 0.1f);
        for (size_t i = 0; i < kCount; i++)
            index.addPoint(v.data(), i);
        index.persistDirty();
    }

    void writeHeaderField(const std::string &dir, std::streamoff offset, size_t value)
    {
        const std::string path = dir + "/header.bin";
        std::fstream header(path.c_str(), std::ios::binary | std::ios::in | std::ios::out);
        if (!header.is_open())
            throw std::runtime_error("test setup: cannot open " + path);
        header.seekp(offset, std::ios::beg);
        if (!header.good())
            throw std::runtime_error("test setup: cannot seek in " + path);
        header.write(reinterpret_cast<const char *>(&value), sizeof(value));
        header.flush();
        if (!header.good())
            throw std::runtime_error("test setup: cannot write to " + path);
    }

    // True only when the load was refused by the layout check. A failed malloc
    // also throws, so matching the reason is what makes this test meaningful:
    // the point is that the allocation is never attempted.
    bool rejectedByLayoutCheck(const std::string &dir)
    {
        hnswlib::InnerProductSpace space(kDim);
        try
        {
            hnswlib::HierarchicalNSW<float> index(&space, dir, false, 0, false, false, true);
        }
        catch (const std::runtime_error &e)
        {
            std::cout << "    rejected: " << e.what() << std::endl;
            // "Cannot load index:" is the layout check. A failed allocation
            // reports "Not enough memory:", which is what unfixed code does and
            // must not be accepted here.
            return std::string(e.what()).find("Cannot load index:") == 0;
        }
        return false;
    }

    void testCraftedSizeDataPerElementIsRejected()
    {
        const std::string dir = ".";

        buildPersistedIndex(dir);
        writeHeaderField(dir, kSizeDataPerElementOffset, static_cast<size_t>(0xFFFFFFFFFFFF0000ULL));
        check(rejectedByLayoutCheck(dir), "wrapping size_data_per_element is refused");

        buildPersistedIndex(dir);
        writeHeaderField(dir, kSizeDataPerElementOffset, static_cast<size_t>(1) << 30);
        check(rejectedByLayoutCheck(dir), "oversized size_data_per_element is refused");

        // maxM0_ is read from the header too, so the recomputation itself must
        // not wrap into a small "expected" value that the header then matches.
        buildPersistedIndex(dir);
        writeHeaderField(dir, kMaxM0Offset, std::numeric_limits<size_t>::max() / 2);
        check(rejectedByLayoutCheck(dir), "wrapping maxM0 is refused");
    }

    void testUntouchedHeaderStillLoads()
    {
        const std::string dir = ".";
        buildPersistedIndex(dir);

        hnswlib::InnerProductSpace space(kDim);
        hnswlib::HierarchicalNSW<float> index(&space, dir, false, 0, false, false, true);
        check(index.cur_element_count == kCount, "an untouched header still loads");
    }
} // namespace

int main()
{
    std::cout << "testCraftedSizeDataPerElementIsRejected" << std::endl;
    testCraftedSizeDataPerElementIsRejected();
    std::cout << "testUntouchedHeaderStillLoads" << std::endl;
    testUntouchedHeaderStillLoads();

    if (failures != 0)
    {
        std::cout << failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "Test persistedHeaderValidation ok" << std::endl;
    return 0;
}

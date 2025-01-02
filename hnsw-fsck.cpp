#include <iostream>
#include "hnswlib/hnswlib.h"

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "USAGE: hnsw-fsck <index_path> <space_name> <dims>\n";
        return 1;
    }
    std::string index_path(argv[1]);
    std::string space_name(argv[2]);
    int dim = atoi(argv[3]);
    std::string index_file = index_path;
    hnswlib::SpaceInterface<float> *l2space;
    bool normalize = false;

    if (space_name == "l2")
    {
        l2space = new hnswlib::L2Space(dim);
        normalize = false;
    }
    else if (space_name == "ip")
    {
        l2space = new hnswlib::InnerProductSpace(dim);
        // For IP, we expect the vectors to be normalized
        normalize = false;
    }
    else if (space_name == "cosine")
    {
        l2space = new hnswlib::InnerProductSpace(dim);
        normalize = true;
    }
    else
    {
        std::cerr << "Unknown space name: " << space_name << std::endl;
        return 2;
    }

    auto appr_alg = new hnswlib::HierarchicalNSW<float>(l2space, index_file, false, 0, false/*allow_replace_deleted*/, normalize, true /*is_persistent_index*/);
    appr_alg->checkIntegrity();
}

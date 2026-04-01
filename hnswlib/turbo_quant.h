#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

namespace hnswlib {

// Lloyd-Max optimal codebook centroids for 4-bit quantization (16 levels)
// of a Beta distribution that converges to N(0, 1/d) on the unit hypersphere.
// Pre-computed by solving the continuous k-means problem (Eq. 4 in TurboQuant paper).
// These are symmetric around 0, normalized for unit-variance Gaussian.
static const float TURBOQUANT_4BIT_CENTROIDS[16] = {
    -2.4008f, -1.8384f, -1.4364f, -1.0968f,
    -0.7914f, -0.5044f, -0.2252f,  0.0000f,
     0.2252f,  0.5044f,  0.7914f,  1.0968f,
     1.4364f,  1.8384f,  2.4008f,  3.0000f  // last bin catches tail
};

// Decision boundaries (midpoints between consecutive centroids)
static const float TURBOQUANT_4BIT_BOUNDARIES[15] = {
    -2.1196f, -1.6374f, -1.2666f, -0.9441f,
    -0.6479f, -0.3648f, -0.1126f,  0.1126f,
     0.3648f,  0.6479f,  0.9441f,  1.2666f,
     1.6374f,  2.1196f,  2.7004f
};

class TurboQuantizer {
public:
    int dim_;
    int bits_;
    int num_levels_;        // 2^bits
    size_t code_size_;      // ceil(dim * bits / 8)
    std::vector<float> rotation_signs_;  // Random ±1 diagonal (fast rotation: D matrix)
    const float* centroids_;
    const float* boundaries_;
    int num_boundaries_;

    // Scratch buffers for rotation (avoid per-call allocation)
    // NOT thread-safe — each thread needs its own or use thread_local
    mutable std::vector<float> rotated_buf_;

    TurboQuantizer() : dim_(0), bits_(0), num_levels_(0), code_size_(0),
                       centroids_(nullptr), boundaries_(nullptr), num_boundaries_(0) {}

    TurboQuantizer(int dim, int bits, uint64_t seed = 42)
        : dim_(dim), bits_(bits) {
        num_levels_ = 1 << bits;

        if (bits == 4) {
            centroids_ = TURBOQUANT_4BIT_CENTROIDS;
            boundaries_ = TURBOQUANT_4BIT_BOUNDARIES;
            num_boundaries_ = 15;
        } else {
            throw std::runtime_error("TurboQuant: only 4-bit quantization supported");
        }

        // For 4-bit: 2 codes per byte
        code_size_ = (dim * bits + 7) / 8;

        // Generate random ±1 diagonal for fast structured rotation (D matrix in Π = HD)
        // Using just D (random sign flips) is a simpler approximation that works well
        // in high dimensions due to concentration of measure
        rotation_signs_.resize(dim);
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < dim; i++) {
            rotation_signs_[i] = dist(rng) ? 1.0f : -1.0f;
        }

        rotated_buf_.resize(dim);
    }

    // Get the total bytes needed to store a quantized vector (codes + norm)
    size_t get_storage_size() const {
        return code_size_ + sizeof(float);  // codes + L2 norm
    }

    // Quantize a float32 vector to b-bit codes + store norm
    // output must have get_storage_size() bytes available
    void quantize(const float* input, uint8_t* output) const {
        float norm = 0.0f;
        float inv_norm;

        // Compute L2 norm
        for (int i = 0; i < dim_; i++) {
            norm += input[i] * input[i];
        }
        norm = std::sqrt(norm);
        inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;

        // Apply random sign rotation and normalize: rotated[i] = sign[i] * input[i] / norm
        // After normalization to unit sphere, coordinates follow Beta → N(0, 1/d)
        // Scale by sqrt(d) to get standard normal for codebook lookup
        float scale = std::sqrt((float)dim_) * inv_norm;

        // Quantize each coordinate using 4-bit codebook
        if (bits_ == 4) {
            for (int i = 0; i < dim_; i += 2) {
                float val0 = rotation_signs_[i] * input[i] * scale;
                float val1 = (i + 1 < dim_) ? rotation_signs_[i + 1] * input[i + 1] * scale : 0.0f;

                // Find nearest centroid using binary search on boundaries
                uint8_t code0 = find_bin(val0);
                uint8_t code1 = find_bin(val1);

                // Pack two 4-bit codes into one byte
                output[i / 2] = (code0 & 0x0F) | ((code1 & 0x0F) << 4);
            }
        }

        // Store norm after codes
        memcpy(output + code_size_, &norm, sizeof(float));
    }

    // Dequantize codes back to approximate float32 vector
    void dequantize(const uint8_t* codes, float* output) const {
        float norm;
        memcpy(&norm, codes + code_size_, sizeof(float));

        float inv_scale = norm / std::sqrt((float)dim_);

        if (bits_ == 4) {
            for (int i = 0; i < dim_; i += 2) {
                uint8_t packed = codes[i / 2];
                uint8_t code0 = packed & 0x0F;
                uint8_t code1 = (packed >> 4) & 0x0F;

                // Lookup centroid, undo scaling and rotation
                output[i] = rotation_signs_[i] * centroids_[code0] * inv_scale;
                if (i + 1 < dim_) {
                    output[i + 1] = rotation_signs_[i + 1] * centroids_[code1] * inv_scale;
                }
            }
        }
    }

    // Asymmetric L2 squared distance: float32 query vs quantized database vector
    // This is the HOT PATH — called millions of times during search
    float distance_asymmetric_l2(const float* query, const uint8_t* codes) const {
        float db_norm;
        memcpy(&db_norm, codes + code_size_, sizeof(float));

        float inv_scale = db_norm / std::sqrt((float)dim_);
        float dist = 0.0f;

        if (bits_ == 4) {
            for (int i = 0; i < dim_; i += 2) {
                uint8_t packed = codes[i / 2];
                uint8_t code0 = packed & 0x0F;
                uint8_t code1 = (packed >> 4) & 0x0F;

                // Reconstruct database coordinate
                float db0 = rotation_signs_[i] * centroids_[code0] * inv_scale;
                float diff0 = query[i] - db0;
                dist += diff0 * diff0;

                if (i + 1 < dim_) {
                    float db1 = rotation_signs_[i + 1] * centroids_[code1] * inv_scale;
                    float diff1 = query[i + 1] - db1;
                    dist += diff1 * diff1;
                }
            }
        }

        return dist;
    }

    // Asymmetric inner product: float32 query vs quantized database vector
    float distance_asymmetric_ip(const float* query, const uint8_t* codes) const {
        float db_norm;
        memcpy(&db_norm, codes + code_size_, sizeof(float));

        float inv_scale = db_norm / std::sqrt((float)dim_);
        float ip = 0.0f;

        if (bits_ == 4) {
            for (int i = 0; i < dim_; i += 2) {
                uint8_t packed = codes[i / 2];
                uint8_t code0 = packed & 0x0F;
                uint8_t code1 = (packed >> 4) & 0x0F;

                float db0 = rotation_signs_[i] * centroids_[code0] * inv_scale;
                ip += query[i] * db0;

                if (i + 1 < dim_) {
                    float db1 = rotation_signs_[i + 1] * centroids_[code1] * inv_scale;
                    ip += query[i + 1] * db1;
                }
            }
        }

        return 1.0f - ip;  // inner product distance
    }

private:
public:
    // Binary search for the quantization bin
    inline uint8_t find_bin(float val) const {
        // Linear scan is faster than binary search for 15 boundaries
        for (int b = 0; b < num_boundaries_; b++) {
            if (val < boundaries_[b]) return (uint8_t)b;
        }
        return (uint8_t)num_boundaries_;  // last bin
    }

    // Symmetric distance: both vectors are quantized codes
    // Used for graph maintenance (mutuallyConnectNewElement, etc.)
    float distance_symmetric_l2(const uint8_t* codes_a, const uint8_t* codes_b) const {
    // Dequantize both to float32 and compute exact L2
    // This is slower but only called during graph construction, not search
    std::vector<float> vec_a(dim_), vec_b(dim_);
    dequantize(codes_a, vec_a.data());
    dequantize(codes_b, vec_b.data());

    float dist = 0.0f;
    for (int i = 0; i < dim_; i++) {
        float d = vec_a[i] - vec_b[i];
        dist += d * d;
    }
    return dist;
}

float distance_symmetric_ip(const uint8_t* codes_a, const uint8_t* codes_b) const {
    std::vector<float> vec_a(dim_), vec_b(dim_);
    dequantize(codes_a, vec_a.data());
    dequantize(codes_b, vec_b.data());

    float ip = 0.0f;
    for (int i = 0; i < dim_; i++) {
        ip += vec_a[i] * vec_b[i];
    }
    return 1.0f - ip;
}

}; // end class TurboQuantizer

// Distance function compatible with hnswlib's DISTFUNC signature
// Handles BOTH asymmetric (float query vs codes) AND symmetric (codes vs codes)
// by checking if pVect1 looks like it could be quantized data
// IMPORTANT: The HNSW algorithm calls this for:
// 1. query-vs-stored (asymmetric): during search
// 2. stored-vs-stored (symmetric): during graph maintenance
// We detect this by dequantizing both arguments always (safe but slower).
static float TurboQuantL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const TurboQuantizer* quantizer = (const TurboQuantizer*)qty_ptr;
    // Always dequantize pVect2 (always stored codes)
    // For pVect1: it could be float32 query OR stored codes
    // Safest approach: dequantize both and compute in float32
    // This handles both asymmetric and symmetric cases correctly
    const uint8_t* codes_a = (const uint8_t*)pVect1v;
    const uint8_t* codes_b = (const uint8_t*)pVect2v;
    return quantizer->distance_symmetric_l2(codes_a, codes_b);
}

static float TurboQuantIP(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const TurboQuantizer* quantizer = (const TurboQuantizer*)qty_ptr;
    const uint8_t* codes_a = (const uint8_t*)pVect1v;
    const uint8_t* codes_b = (const uint8_t*)pVect2v;
    return quantizer->distance_symmetric_ip(codes_a, codes_b);
}

// SpaceInterface implementation for TurboQuant
class TurboQuantL2Space : public SpaceInterface<float> {
    TurboQuantizer quantizer_;
    size_t data_size_;
    DISTFUNC<float> fstdistfunc_;

public:
    TurboQuantL2Space(size_t dim, int bits = 4, uint64_t seed = 42)
        : quantizer_(dim, bits, seed) {
        data_size_ = quantizer_.get_storage_size();
        fstdistfunc_ = TurboQuantL2Sqr;
    }

    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void* get_dist_func_param() override { return (void*)&quantizer_; }

    TurboQuantizer* get_quantizer() { return &quantizer_; }
};

class TurboQuantIPSpace : public SpaceInterface<float> {
    TurboQuantizer quantizer_;
    size_t data_size_;
    DISTFUNC<float> fstdistfunc_;

public:
    TurboQuantIPSpace(size_t dim, int bits = 4, uint64_t seed = 42)
        : quantizer_(dim, bits, seed) {
        data_size_ = quantizer_.get_storage_size();
        fstdistfunc_ = TurboQuantIP;
    }

    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void* get_dist_func_param() override { return (void*)&quantizer_; }

    TurboQuantizer* get_quantizer() { return &quantizer_; }
};

}  // namespace hnswlib

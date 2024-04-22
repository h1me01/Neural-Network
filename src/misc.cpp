#include "misc.h"

namespace Tools {
    random_device rd;
    mt19937 gen(42);
} // namespace Tools

int popLsb(uint64_t &b) {
    int lsb = bsf(b);
    b &= b - 1;
    return lsb;
}

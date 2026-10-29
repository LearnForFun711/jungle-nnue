#include <cstdio>
#include "../src/jungle.hpp"
using namespace jungle;
int main() {
    const Terrain& t = terrain();
    std::printf("den_owner (RED=0, BLACK=1, -1 none):\n");
    for (int r = 0; r < HEIGHT; ++r) {
        for (int c = 0; c < WIDTH; ++c) {
            const int s = sq_index(r, c);
            if (t.den_owner[s] >= 0)
                std::printf("  square %2d = (r%d,c%d) owner %d\n", s, r, c, t.den_owner[s]);
        }
    }
    std::printf("square 12 = (r%d,c%d) den_owner %d\n", 12/WIDTH, 12%WIDTH, t.den_owner[12]);
    std::printf("square 59 = (r%d,c%d) den_owner %d\n", 59/WIDTH, 59%WIDTH, t.den_owner[59]);
    return 0;
}

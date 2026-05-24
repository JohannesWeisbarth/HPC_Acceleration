#include <Kokkos_Core.hpp>
#include "streaming.hpp"

// Grid sizes in x and y direction.
const size_t X = 15;
const size_t Y = 10;
const size_t I = 9;

// The 9 D2Q9 directions.
constexpr int d2q9_directions[9][2] = {
    {0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}
};

// AI generated
#include <fstream>
void write_output(const Kokkos::View<double**> &rho, const Kokkos::View<double***> &velocities, int timestep) {
    auto rho_h = Kokkos::create_mirror_view(rho);
    auto velocities_h = Kokkos::create_mirror_view(velocities);
    Kokkos::deep_copy(rho_h, rho);
    Kokkos::deep_copy(velocities_h, velocities);

    std::ofstream file("output_" + std::to_string(timestep) + ".csv");
    file << "x,y,rho,vx,vy\n";
    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y)
            file << x << "," << y << ","
                 << rho_h(x,y) << ","
                 << velocities_h(x,y,0) << ","
                 << velocities_h(x,y,1) << "\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // The distribution function.
        auto f = Kokkos::View<double***>("f", X, Y, I);
        // The density vector.
        auto rho = Kokkos::View<double**>("density", X, Y);
        // The velocities in x and y directions.
        auto velocities = Kokkos::View<double***>("velocities", X, Y, 2);

        initialize(f);

        // Test run a few times.
        for (int t = 0; t < 40; ++t) {
            streaming(f);
            compute_density(f, rho);
            compute_velocity(f, rho, velocities);

            write_output(rho, velocities, t);
        }
    }
    Kokkos::finalize();
    return 0;
}

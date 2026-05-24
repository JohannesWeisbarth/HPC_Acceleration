#include <streaming.hpp>
#include <Kokkos_Core.hpp>

// Grid sizes in x and y direction.
const size_t X = 15;
const size_t Y = 10;
const size_t I = 9;

// The 9 D2Q9 directions.
constexpr int d2q9_directions[9][2] = {
    {0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}
};

// The distribution function.
auto f = Kokkos::View<double***>("f", X, Y, I);
// The density vector.
auto rho = Kokkos::View<double**>("density", X, Y);
// The velocities in x and y directions.
auto velocities = Kokkos::View<double***>("velocities", X, Y, 2);

void initialize(const Kokkos::View<double***> &f) {
    // Init densities with 0.
    Kokkos::deep_copy(f, 0.0);
    // Add a particle in the center:
    f(X / 2, Y / 2, 0) = 1.0;
}

void compute_density(
    const Kokkos::View<double***> &f,
    const Kokkos::View<double**> &rho
) {
    for (size_t i = 0; i < X; ++i) {
        for (size_t j = 0; j < Y; ++j) {
            double sum = 0.0;
            // Sum over the velocity directions.
            for (size_t k = 0; k < I; ++k) {
                sum += f(i, j, k);
            }
            rho(i, j) = sum;
        }
    }
}

void compute_velocity(
    const Kokkos::View<double***> &f,
    const Kokkos::View<double**> &rho,
    const Kokkos::View<double***> &velocities
) {
    for (size_t i = 0; i < X; ++i) {
        for (size_t j = 0; j < Y; ++j) {
            double v_x = 0.0;
            double v_y = 0.0;

            // Compute velocity in x and y direction.
            // The d2q9_directions are the c_i.
            for (size_t k = 0; k < I; ++k) {
                v_x += f(i, j, k) * d2q9_directions[k][0];
                v_y += f(i, j, k) * d2q9_directions[k][1];
            }

            // Avoid division by zero.
            if (rho(i, j) > 0.0) {
                v_x /= rho(i, j);
                v_y /= rho(i, j);
            } else {
                v_x = 0.0;
                v_y = 0.0;
            }

            // Update velocities.
            velocities(i, j, 0) = v_x;
            velocities(i, j, 1) = v_y;
        }
    }
}

void streaming(const Kokkos::View<double***> &f) {
    // Save old values.
    auto f_old = Kokkos::View<double***>("f_old", X, Y, I);
    Kokkos::deep_copy(f_old, f);

    Kokkos::parallel_for("streaming", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {X, Y, I}),
        KOKKOS_LAMBDA(int x, int y, int i) {
            // Compute source position. Minus must be used to ensure the correct streaming direction!
            int src_x = (x - d2q9_directions[i][0] + X) % X;
            int src_y = (y - d2q9_directions[i][1] + Y) % Y;
            f(x, y, i) = f_old(src_x, src_y, i);
        }
    );
}

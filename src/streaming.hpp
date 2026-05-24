#ifndef STREAMING_H
#define STREAMING_H

#include <Kokkos_Core.hpp>

void initialize(const Kokkos::View<double***> &f);

void compute_density(
    const Kokkos::View<double***> &f,
    const Kokkos::View<double**> &rho
);

void compute_velocity(
    const Kokkos::View<double***> &f,
    const Kokkos::View<double**> &rho,
    const Kokkos::View<double***> &velocities
);

void streaming(const Kokkos::View<double***> &f);

#endif
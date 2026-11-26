#pragma once

#include "fedspatial/model.hpp"

#include <filesystem>
#include <vector>

namespace fedspatial {

// Exact FP64 inputs exported by tests/export_figure1_python_fixture.py.
// Workers are formed after loading by the original consecutive torch.chunk
// partition, never by creating separate worker processes.
struct Figure1Fixture {
  Matrix sites, knots, x, latent, noise, z;
};

bool load_figure1_python_fixture(const std::filesystem::path& directory,
                                 Figure1Fixture* fixture);
std::vector<WorkerInput> partition_figure1_fixture(const Figure1Fixture& fixture,
                                                    int workers = 10);

}  // namespace fedspatial

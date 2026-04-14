#include "model_io.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x4C474E4E;  // LGNN
constexpr uint32_t kVersion = 1;

void write_u32(std::ofstream& out, uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void write_u64(std::ofstream& out, uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
uint32_t read_u32(std::ifstream& in) {
  uint32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}
uint64_t read_u64(std::ifstream& in) {
  uint64_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}
}  // namespace

void save_model_binary(const std::shared_ptr<Module>& model, const std::string& path) {
  if (!model) throw std::runtime_error("save_model_binary expects a valid model");
  auto params = model->parameters();
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("save_model_binary failed to open output file");

  write_u32(out, kMagic);
  write_u32(out, kVersion);
  write_u64(out, static_cast<uint64_t>(params.size()));
  for (const auto& p : params) {
    const auto t = p.data();
    const auto dims = t.get_dims();
    const auto data = t.get_data();
    write_u64(out, static_cast<uint64_t>(dims.size()));
    for (size_t d : dims) write_u64(out, static_cast<uint64_t>(d));
    write_u64(out, static_cast<uint64_t>(data.size()));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(double)));
  }
  if (!out.good()) throw std::runtime_error("save_model_binary failed while writing checkpoint");
}

void load_model_binary(const std::shared_ptr<Module>& model, const std::string& path) {
  if (!model) throw std::runtime_error("load_model_binary expects a valid model");
  auto params = model->parameters();
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("load_model_binary failed to open input file");

  const uint32_t magic = read_u32(in);
  const uint32_t version = read_u32(in);
  if (magic != kMagic) throw std::runtime_error("load_model_binary invalid checkpoint magic");
  if (version != kVersion) throw std::runtime_error("load_model_binary unsupported checkpoint version");
  const uint64_t n_params = read_u64(in);
  if (n_params != static_cast<uint64_t>(params.size())) {
    throw std::runtime_error("load_model_binary parameter count mismatch");
  }

  struct Blob {
    std::vector<size_t> dims;
    std::vector<double> data;
  };
  std::vector<Blob> blobs(static_cast<size_t>(n_params));

  for (size_t i = 0; i < blobs.size(); ++i) {
    const uint64_t nd = read_u64(in);
    blobs[i].dims.resize(static_cast<size_t>(nd));
    for (size_t j = 0; j < blobs[i].dims.size(); ++j) {
      blobs[i].dims[j] = static_cast<size_t>(read_u64(in));
    }
    const uint64_t count = read_u64(in);
    blobs[i].data.resize(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(blobs[i].data.data()), static_cast<std::streamsize>(count * sizeof(double)));
  }
  if (!in.good() && !in.eof()) throw std::runtime_error("load_model_binary failed while reading checkpoint");

  for (size_t i = 0; i < params.size(); ++i) {
    const auto cur = params[i].data();
    if (cur.get_dims() != blobs[i].dims) {
      throw std::runtime_error("load_model_binary tensor shape mismatch");
    }
  }
  for (size_t i = 0; i < params.size(); ++i) {
    const auto cur = params[i].data();
    Tensor restored = Tensor::from_data(blobs[i].dims, blobs[i].data, cur.get_device_type(), cur.get_device_index());
    params[i].set_data(restored);
    params[i].zero_grad();
  }
}

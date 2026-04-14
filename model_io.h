#pragma once

#include <memory>
#include <string>

#include "layers.h"

// Binary (non-pickle) model-only checkpoint helpers for non-Python bindings.
void save_model_binary(const std::shared_ptr<Module>& model, const std::string& path);
void load_model_binary(const std::shared_ptr<Module>& model, const std::string& path);

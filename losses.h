#pragma once

#include "autograd.h"

inline Variable mse_loss(const Variable& pred, const Variable& target) {
  return Variable::mse_loss(pred, target);
}

// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <boost/lockfree/stack.hpp>
#include <cstddef>
#include <utility>

namespace opensn
{

/// Recyclable lock-free stack with multi-producer push and single-consumer drain.
template <class T>
class LockFreeTreiberStack
{
public:
  LockFreeTreiberStack() : stack_(0) {}

  void Preallocate(std::size_t count)
  {
    if (count == 0)
      return;

    stack_.reserve_unsafe(count);
  }

  void Push(T&& payload) { stack_.push(std::move(payload)); }

  template <class F>
  bool DrainAndProcess(F&& callback)
  {
    return stack_.consume_all_atomic(
             [&callback](T&& payload) { callback(std::move(payload)); }) > 0;
  }

  bool Empty() const { return stack_.empty(); }

private:
  boost::lockfree::stack<T> stack_;
};

} // namespace opensn

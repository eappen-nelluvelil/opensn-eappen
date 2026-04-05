// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
// #include "framework/logging/log.h"
// #include "framework/mesh/mesh_continuum/mesh_continuum.h"
// #include "framework/runtime.h"
// #include "caliper/cali.h"
// #include <queue>
// #include <limits>
// #include <algorithm>
// #include <cstring>
// #include <numeric>
// #include <stdexcept>
// #include <boost/graph/topological_sort.hpp>

// #if __AVX512F__ || __AVX2__
// #include <immintrin.h>
// #endif

// namespace opensn
// {

// namespace
// {

// constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// class AlignedBitset
// {
// public:
// #if __AVX512F__
//   static constexpr std::size_t kWordsAlign = 8;
// #elif __AVX2__
//   static constexpr std::size_t kWordsAlign = 4;
// #else
//   static constexpr std::size_t kWordsAlign = 1;
// #endif

//   explicit AlignedBitset(std::size_t n)
//     : n_(n),
//       words_(((n + 63) / 64 + kWordsAlign - 1) / kWordsAlign * kWordsAlign),
//       data_(words_, 0ULL)
//   {
//   }

//   void Clear() { std::fill(data_.begin(), data_.end(), 0ULL); }
  
//   void SetBit(std::size_t i) { data_[i / 64] |= (1ULL << (i % 64)); }
//   void ClearBit(std::size_t i) { data_[i / 64] &= ~(1ULL << (i % 64)); }
//   bool TestBit(std::size_t i) const { return data_[i / 64] & (1ULL << (i % 64)); }

//   void CopyFrom(const AlignedBitset& other)
//   {
//     std::memcpy(data_.data(), other.data_.data(), words_ * sizeof(std::uint64_t));
//   }

//   void OrWith(const AlignedBitset& other)
//   {
//     std::uint64_t* d = data_.data();
//     const std::uint64_t* s = other.data_.data();
// #if __AVX512F__
//     for (std::size_t w = 0; w < words_; w += 8)
//     {
//       __m512i vd = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(d + w));
//       __m512i vs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + w));
//       _mm512_storeu_si512(reinterpret_cast<__m512i*>(d + w), _mm512_or_si512(vd, vs));
//     }
// #elif __AVX2__
//     for (std::size_t w = 0; w < words_; w += 4)
//     {
//       __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
//       __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
//       _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w), _mm256_or_si256(vd, vs));
//     }
// #else
//     for (std::size_t w = 0; w < words_; ++w)
//       d[w] |= s[w];
// #endif
//   }

//   void AndWith(const AlignedBitset& other)
//   {
//     std::uint64_t* d = data_.data();
//     const std::uint64_t* s = other.data_.data();
// #if __AVX512F__
//     for (std::size_t w = 0; w < words_; w += 8)
//     {
//       __m512i vd = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(d + w));
//       __m512i vs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + w));
//       _mm512_storeu_si512(reinterpret_cast<__m512i*>(d + w), _mm512_and_si512(vd, vs));
//     }
// #elif __AVX2__
//     for (std::size_t w = 0; w < words_; w += 4)
//     {
//       __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
//       __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
//       _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w), _mm256_and_si256(vd, vs));
//     }
// #else
//     for (std::size_t w = 0; w < words_; ++w)
//       d[w] &= s[w];
// #endif
//   }

//   std::size_t FindFirstSet(std::size_t start_pos = 0) const
//   {
//     std::size_t w = start_pos / 64;
//     if (w >= words_)
//       return n_;

//     std::uint64_t masked = data_[w] & (~0ULL << (start_pos % 64));
//     if (masked)
//     {
//       std::size_t pos = w * 64 + static_cast<std::size_t>(__builtin_ctzll(masked));
//       return pos < n_ ? pos : n_;
//     }

//     for (++w; w < words_; ++w)
//     {
//       if (data_[w])
//       {
//         std::size_t pos = w * 64 + static_cast<std::size_t>(__builtin_ctzll(data_[w]));
//         return pos < n_ ? pos : n_;
//       }
//     }
//     return n_;
//   }

//   std::size_t FindNextSet(std::size_t pos) const { return FindFirstSet(pos + 1); }

// private:
//   std::size_t n_;
//   std::size_t words_;
//   std::vector<std::uint64_t> data_;
// };

// class OnTheFlyHopcroftKarp
// {
// public:
//   static constexpr std::size_t kMaxDFSDepth = 64;

//   OnTheFlyHopcroftKarp(std::uint32_t num_tasks,
//                        const std::vector<Task>& task_list,
//                        std::vector<std::uint32_t>& task_slot_ids)
//     : num_tasks_(num_tasks),
//       task_list_(task_list),
//       task_slot_ids_(task_slot_ids),
//       mate_u_(num_tasks, INVALID_INDEX),
//       mate_v_(num_tasks, INVALID_INDEX),
//       dist_(num_tasks, -1),
//       bfs_buffer_(num_tasks_),
//       dfs_pool_(kMaxDFSDepth, AlignedBitset(num_tasks_)),
//       queue_(num_tasks)
//   {
//   }

//   std::size_t Solve()
//   {
//     std::size_t matching_size = GreedyInit();
//     while (BFS())
//     {
//       for (std::uint32_t u = 0; u < num_tasks_; ++u)
//       {
//         if (mate_u_[u] == INVALID_INDEX && DFS(u, 0))
//           ++matching_size;
//       }
//     }
    
//     AssignStaticSlots();
//     return static_cast<std::size_t>(num_tasks_) - matching_size;
//   }

// private:
//   void ComputeDescendantsBFS(std::uint32_t start, AlignedBitset& result)
//   {
//     result.Clear();
//     result.SetBit(start);

//     std::size_t head = 0, tail = 0;
//     queue_[tail++] = start;

//     while (head < tail)
//     {
//       std::uint32_t v = queue_[head++];
//       for (const auto& s : task_list_[v].successors)
//       {
//         if (!result.TestBit(s))
//         {
//           result.SetBit(s);
//           queue_[tail++] = s;
//         }
//       }
//     }
//   }

//   void ComputeReuseTargets(std::uint32_t u, AlignedBitset& result)
//   {
//     const auto& task = task_list_[u];
//     if (task.successors.empty())
//     {
//       result.Clear();
//       return;
//     }

//     ComputeDescendantsBFS(task.successors[0], result);

//     AlignedBitset temp_buf(num_tasks_);
//     for (std::size_t i = 1; i < task.successors.size(); ++i)
//     {
//       ComputeDescendantsBFS(task.successors[i], temp_buf);
//       result.AndWith(temp_buf);
//     }

//     for (const auto& succ : task.successors)
//       result.ClearBit(succ);
//   }

//   std::size_t GreedyInit()
//   {
//     std::size_t count = 0;
//     for (std::uint32_t u = 0; u < num_tasks_; ++u)
//     {
//       if (mate_u_[u] != INVALID_INDEX)
//         continue;
        
//       ComputeReuseTargets(u, bfs_buffer_);
//       std::size_t v = bfs_buffer_.FindFirstSet();
//       while (v < num_tasks_)
//       {
//         if (mate_v_[v] == INVALID_INDEX)
//         {
//           mate_u_[u] = static_cast<std::uint32_t>(v);
//           mate_v_[v] = u;
//           ++count;
//           break;
//         }
//         v = bfs_buffer_.FindNextSet(v);
//       }
//     }
//     return count;
//   }

//   bool BFS()
//   {
//     std::fill(dist_.begin(), dist_.end(), -1);
//     std::size_t head = 0, tail = 0;

//     for (std::uint32_t u = 0; u < num_tasks_; ++u)
//     {
//       if (mate_u_[u] == INVALID_INDEX)
//       {
//         dist_[u] = 0;
//         queue_[tail++] = u;
//       }
//     }

//     dist_null_ = std::numeric_limits<int>::max();

//     while (head < tail)
//     {
//       std::uint32_t u = queue_[head++];

//       if (dist_[u] < dist_null_)
//       {
//         ComputeReuseTargets(u, bfs_buffer_);
//         std::size_t v = bfs_buffer_.FindFirstSet();
        
//         while (v < num_tasks_)
//         {
//           std::uint32_t mate_of_v = mate_v_[v];
//           if (mate_of_v == INVALID_INDEX)
//           {
//             if (dist_null_ == std::numeric_limits<int>::max())
//               dist_null_ = dist_[u] + 1;
//           }
//           else if (dist_[mate_of_v] == -1)
//           {
//             dist_[mate_of_v] = dist_[u] + 1;
//             queue_[tail++] = mate_of_v;
//           }
//           v = bfs_buffer_.FindNextSet(v);
//         }
//       }
//     }
//     return dist_null_ != std::numeric_limits<int>::max();
//   }

//   bool DFS(std::uint32_t u, std::size_t depth)
//   {
//     if (depth >= kMaxDFSDepth)
//       return false;

//     AlignedBitset& neighbors = dfs_pool_[depth];
//     ComputeReuseTargets(u, neighbors);

//     std::size_t v = neighbors.FindFirstSet();
//     while (v < num_tasks_)
//     {
//       std::uint32_t mate_of_v = mate_v_[v];
//       if (mate_of_v == INVALID_INDEX)
//       {
//         if (dist_null_ == dist_[u] + 1)
//         {
//           mate_v_[v] = u;
//           mate_u_[u] = static_cast<std::uint32_t>(v);
//           dist_[u] = -1;
//           return true;
//         }
//       }
//       else if (dist_[mate_of_v] == dist_[u] + 1)
//       {
//         if (DFS(mate_of_v, depth + 1))
//         {
//           mate_v_[v] = u;
//           mate_u_[u] = static_cast<std::uint32_t>(v);
//           dist_[u] = -1;
//           return true;
//         }
//       }
//       v = neighbors.FindNextSet(v);
//     }
//     dist_[u] = -1;
//     return false;
//   }

//   void AssignStaticSlots()
//   {
//     task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
//     std::uint32_t next_slot_id = 0;

//     for (std::uint32_t task_id = 0; task_id < num_tasks_; ++task_id)
//     {
//       if (mate_v_[task_id] == INVALID_INDEX)
//       {
//         std::uint32_t current = task_id;
//         while (current != INVALID_INDEX)
//         {
//           task_slot_ids_[current] = next_slot_id;
//           current = mate_u_[current];
//         }
//         ++next_slot_id;
//       }
//     }
//   }

//   std::uint32_t num_tasks_;
//   const std::vector<Task>& task_list_;
//   std::vector<std::uint32_t>& task_slot_ids_;

//   std::vector<std::uint32_t> mate_u_;
//   std::vector<std::uint32_t> mate_v_;
//   std::vector<int> dist_;
//   int dist_null_ = 0;
  
//   AlignedBitset bfs_buffer_;
//   std::vector<AlignedBitset> dfs_pool_;
//   std::vector<std::uint32_t> queue_;
// };

// } // namespace

// void
// CBC_SPDS::BuildTaskGraph()
// {
//   constexpr auto INCOMING = FaceOrientation::INCOMING;
//   constexpr auto OUTGOING = FaceOrientation::OUTGOING;

//   const auto num_loc_cells = grid_->local_cells.size();
//   task_list_.assign(num_loc_cells, Task{});
//   local_successor_offsets_.resize(num_loc_cells + 1, 0);

//   std::size_t successor_count = 0;
//   for (const auto& cell : grid_->local_cells)
//   {
//     unsigned int num_dependencies = 0;
//     std::vector<std::uint32_t> predecessors;
//     std::vector<std::uint32_t> successors;

//     for (std::size_t f = 0; f < cell.faces.size(); ++f)
//     {
//       const auto& face = cell.faces[f];
//       const auto orientation = cell_face_orientations_[cell.local_id][f];

//       if (orientation == INCOMING and face.has_neighbor)
//       {
//         ++num_dependencies;
//         if (face.IsNeighborLocal(grid_.get()))
//           predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
//       }
//       else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
//         successors.push_back(grid_->cells[face.neighbor_id].local_id);
//     }

//     successor_count += successors.size();
//     local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);
//     task_list_[cell.local_id] = Task{
//       0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
//   }

//   local_successors_.resize(successor_count);
//   initial_successors_to_retire_.resize(task_list_.size());
//   for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
//   {
//     initial_successors_to_retire_[cell_id] =
//       static_cast<std::uint32_t>(task_list_[cell_id].successors.size());
//     std::copy(task_list_[cell_id].successors.begin(),
//               task_list_[cell_id].successors.end(),
//               local_successors_.begin() + local_successor_offsets_[cell_id]);
//   }
// }

// CBC_SPDS::CBC_SPDS(const Vector3& omega,
//                    const std::shared_ptr<MeshContinuum>& grid,
//                    const bool allow_cycles)
//   : SPDS(omega, grid)
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

//   const auto num_loc_cells = grid->local_cells.size();

//   std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
//   std::set<int> location_successors;
//   std::set<int> location_dependencies;

//   PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

//   location_successors_.reserve(location_successors.size());
//   location_dependencies_.reserve(location_dependencies.size());

//   for (const auto loc : location_successors)
//     location_successors_.push_back(loc);

//   for (const auto loc : location_dependencies)
//     location_dependencies_.push_back(loc);

//   Graph local_dg(num_loc_cells);
//   for (std::size_t c = 0; c < num_loc_cells; ++c)
//     for (const auto& successor : cell_successors[c])
//       boost::add_edge(c, successor.first, successor.second, local_dg);

//   if (allow_cycles)
//   {
//     const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
//     for (const auto& [u, v] : edges_to_remove)
//       local_sweep_fas_.emplace_back(u, v);
//   }

//   spls_.clear();
//   boost::topological_sort(local_dg, std::back_inserter(spls_));
//   std::reverse(spls_.begin(), spls_.end());
//   if (spls_.empty())
//   {
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
//                            "Cycles need to be allowed by the calling application.");
//   }

//   topo_order_.reserve(spls_.size());
//   for (const auto v : spls_)
//     topo_order_.push_back(static_cast<std::uint32_t>(v));

//   std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
//   CommunicateLocationDependencies(location_dependencies_, global_dependencies);
//   BuildTaskGraph();

//   max_num_local_psi_slots_ = num_loc_cells;
//   task_slot_ids_.resize(num_loc_cells);
//   std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
// }

// const std::vector<Task>&
// CBC_SPDS::GetTaskList() const noexcept
// {
//   return task_list_;
// }

// void
// CBC_SPDS::ComputeMaxNumLocalPsiSlots()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

//   const std::uint32_t num_tasks = static_cast<std::uint32_t>(task_list_.size());
//   if (num_tasks == 0)
//   {
//     max_num_local_psi_slots_ = 0;
//     return;
//   }

//   OnTheFlyHopcroftKarp allocator(num_tasks, task_list_, task_slot_ids_);
//   max_num_local_psi_slots_ = allocator.Solve();
// }

// #ifndef __OPENSN_WITH_GPU__
// void
// CBC_SPDS::CopyTaskGraphDataOnDevice() const
// {
// }

// void
// CBC_SPDS::FreeDeviceData() const
// {
// }
// #endif

// CBC_SPDS::~CBC_SPDS()
// {
//   FreeDeviceData();
// }

// } // namespace opensn

// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
// #include "framework/logging/log.h"
// #include "framework/mesh/mesh_continuum/mesh_continuum.h"
// #include "framework/runtime.h"
// #include "caliper/cali.h"
// #include <queue>
// #include <limits>
// #include <algorithm>
// #include <cstring>
// #include <numeric>
// #include <stdexcept>
// #include <boost/graph/topological_sort.hpp>

// #if __AVX512F__ || __AVX2__
// #include <immintrin.h>
// #endif

// namespace opensn
// {

// namespace
// {

// constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// class AVXBitMatrix
// {
// public:
// #if __AVX512F__
//   static constexpr std::size_t kWordsAlign = 8;
// #elif __AVX2__
//   static constexpr std::size_t kWordsAlign = 4;
// #else
//   static constexpr std::size_t kWordsAlign = 1;
// #endif

//   AVXBitMatrix() : n_(0), words_per_row_(0) {}

//   void ResizeAndClear(std::size_t n)
//   {
//     n_ = n;
//     words_per_row_ = ((n + 63) / 64 + kWordsAlign - 1) / kWordsAlign * kWordsAlign;
//     const std::size_t required_words = n * words_per_row_;
    
//     if (data_.size() < required_words)
//       data_.resize(required_words);
      
//     std::memset(data_.data(), 0, required_words * sizeof(std::uint64_t));
//   }

//   std::size_t WordsPerRow() const { return words_per_row_; }
//   std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }
//   const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }
  
//   void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }
//   void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

//   void CopyRow(std::size_t dst, const AVXBitMatrix& src_mat, std::size_t src_row)
//   {
//     std::memcpy(Row(dst), src_mat.Row(src_row), words_per_row_ * sizeof(std::uint64_t));
//   }

//   void ClearRow(std::size_t i)
//   {
//     std::memset(Row(i), 0, words_per_row_ * sizeof(std::uint64_t));
//   }

//   void OrRows(std::size_t dst, const AVXBitMatrix& src_mat, std::size_t src_row)
//   {
//     std::uint64_t* d = Row(dst);
//     const std::uint64_t* s = src_mat.Row(src_row);
// #if __AVX512F__
//     for (std::size_t w = 0; w < words_per_row_; w += 8)
//     {
//       __m512i vd = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(d + w));
//       __m512i vs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + w));
//       _mm512_storeu_si512(reinterpret_cast<__m512i*>(d + w), _mm512_or_si512(vd, vs));
//     }
// #elif __AVX2__
//     for (std::size_t w = 0; w < words_per_row_; w += 4)
//     {
//       __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
//       __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
//       _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w), _mm256_or_si256(vd, vs));
//     }
// #else
//     for (std::size_t w = 0; w < words_per_row_; ++w) d[w] |= s[w];
// #endif
//   }

//   void AndRows(std::size_t dst, const AVXBitMatrix& src_mat, std::size_t src_row)
//   {
//     std::uint64_t* d = Row(dst);
//     const std::uint64_t* s = src_mat.Row(src_row);
// #if __AVX512F__
//     for (std::size_t w = 0; w < words_per_row_; w += 8)
//     {
//       __m512i vd = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(d + w));
//       __m512i vs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + w));
//       _mm512_storeu_si512(reinterpret_cast<__m512i*>(d + w), _mm512_and_si512(vd, vs));
//     }
// #elif __AVX2__
//     for (std::size_t w = 0; w < words_per_row_; w += 4)
//     {
//       __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
//       __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
//       _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w), _mm256_and_si256(vd, vs));
//     }
// #else
//     for (std::size_t w = 0; w < words_per_row_; ++w) d[w] &= s[w];
// #endif
//   }

//   std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
//   {
//     const std::uint64_t* r = Row(row);
//     std::size_t w = start_pos / 64;

//     if (w >= words_per_row_) return n_;

//     // Handle partial first word
//     std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));
//     if (masked)
//       return std::min(n_, w * 64 + static_cast<std::size_t>(__builtin_ctzll(masked)));

//     ++w;

//     // SIMD Accelerated Block Scanning skips arrays of 0s instantly
// #if __AVX512F__
//     const __m512i v_zero = _mm512_setzero_si512();
//     for (; w + 7 < words_per_row_; w += 8)
//     {
//       __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r + w));
//       if (_mm512_cmpneq_epi64_mask(v, v_zero) != 0) break;
//     }
// #elif __AVX2__
//     const __m256i v_zero = _mm256_setzero_si256();
//     for (; w + 3 < words_per_row_; w += 4)
//     {
//       __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + w));
//       if (!_mm256_testz_si256(v, v)) break;
//     }
// #endif

//     // Scalar fallback/tail processing
//     for (; w < words_per_row_; ++w)
//     {
//       if (r[w])
//         return std::min(n_, w * 64 + static_cast<std::size_t>(__builtin_ctzll(r[w])));
//     }
//     return n_;
//   }

//   std::size_t FindNextSet(std::size_t row, std::size_t pos) const 
//   { 
//     return FindFirstSet(row, pos + 1); 
//   }

// private:
//   std::size_t n_;
//   std::size_t words_per_row_;
//   std::vector<std::uint64_t> data_;
// };

// // Thread-local workspace prevents massive dynamic allocation overhead and 
// // locks memory to the thread's executing NUMA node.
// struct ThreadLocalWorkspace
// {
//   AVXBitMatrix reachability;
//   AVXBitMatrix reuse_targets;
//   std::vector<std::uint32_t> mate_u;
//   std::vector<std::uint32_t> mate_v;
//   std::vector<int> dist;
//   std::vector<std::uint32_t> queue;
  
//   void Prepare(std::size_t n)
//   {
//     reachability.ResizeAndClear(n);
//     reuse_targets.ResizeAndClear(n);
//     mate_u.assign(n, INVALID_INDEX);
//     mate_v.assign(n, INVALID_INDEX);
//     dist.assign(n, -1);
//     if (queue.size() < n) queue.resize(n);
//   }
// };

// class DenseHopcroftKarp
// {
// public:
//   DenseHopcroftKarp(std::uint32_t num_tasks,
//                     const std::vector<Task>& task_list,
//                     const std::vector<std::uint32_t>& topo_order,
//                     std::vector<std::uint32_t>& task_slot_ids,
//                     ThreadLocalWorkspace& ws)
//     : num_tasks_(num_tasks),
//       task_list_(task_list),
//       topo_order_(topo_order),
//       task_slot_ids_(task_slot_ids),
//       ws_(ws)
//   {
//     ws_.Prepare(num_tasks_);
//   }

//   std::size_t Solve()
//   {
//     // 1. Build Transitive Closure Bottom-Up (Projecting downstream dependencies up)
//     for (auto it = topo_order_.rbegin(); it != topo_order_.rend(); ++it)
//     {
//       const std::uint32_t u = *it;
//       ws_.reachability.SetBit(u, u);
//       for (const auto& succ : task_list_[u].successors)
//         ws_.reachability.OrRows(u, ws_.reachability, succ); 
//     }

//     // 2. Precompute Exact Reuse Intersections
//     // R_U = Intersect(Desc(S)) \ succ(U) for S in succ(U)
//     for (std::uint32_t u = 0; u < num_tasks_; ++u)
//     {
//       const auto& successors = task_list_[u].successors;
//       if (successors.empty()) continue;

//       // Draw sets from the constructed reachability graph
//       ws_.reuse_targets.CopyRow(u, ws_.reachability, successors[0]);
//       for (std::size_t i = 1; i < successors.size(); ++i)
//         ws_.reuse_targets.AndRows(u, ws_.reachability, successors[i]);

//       for (const auto& succ : successors)
//         ws_.reuse_targets.ClearBit(u, succ);
//     }

//     // 3. Dense Hopcroft-Karp Bipartite Matching
//     std::size_t matching_size = GreedyInit();
//     while (BFS())
//     {
//       for (std::uint32_t u = 0; u < num_tasks_; ++u)
//       {
//         if (ws_.mate_u[u] == INVALID_INDEX && DFS(u))
//           ++matching_size;
//       }
//     }
    
//     AssignStaticSlots();
//     return static_cast<std::size_t>(num_tasks_) - matching_size;
//   }

// private:
//   std::size_t GreedyInit()
//   {
//     std::size_t count = 0;
//     for (std::uint32_t u = 0; u < num_tasks_; ++u)
//     {
//       if (ws_.mate_u[u] != INVALID_INDEX) continue;
        
//       std::size_t v = ws_.reuse_targets.FindFirstSet(u);
//       while (v < num_tasks_)
//       {
//         if (ws_.mate_v[v] == INVALID_INDEX)
//         {
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.mate_v[v] = u;
//           ++count;
//           break;
//         }
//         v = ws_.reuse_targets.FindNextSet(u, v);
//       }
//     }
//     return count;
//   }

//   bool BFS()
//   {
//     std::fill(ws_.dist.begin(), ws_.dist.begin() + num_tasks_, -1);
//     std::size_t head = 0, tail = 0;

//     for (std::uint32_t u = 0; u < num_tasks_; ++u)
//     {
//       if (ws_.mate_u[u] == INVALID_INDEX)
//       {
//         ws_.dist[u] = 0;
//         ws_.queue[tail++] = u;
//       }
//     }

//     dist_null_ = std::numeric_limits<int>::max();

//     while (head < tail)
//     {
//       std::uint32_t u = ws_.queue[head++];

//       if (ws_.dist[u] < dist_null_)
//       {
//         std::size_t v = ws_.reuse_targets.FindFirstSet(u);
//         while (v < num_tasks_)
//         {
//           std::uint32_t mate_of_v = ws_.mate_v[v];
//           if (mate_of_v == INVALID_INDEX)
//           {
//             if (dist_null_ == std::numeric_limits<int>::max())
//               dist_null_ = ws_.dist[u] + 1;
//           }
//           else if (ws_.dist[mate_of_v] == -1)
//           {
//             ws_.dist[mate_of_v] = ws_.dist[u] + 1;
//             ws_.queue[tail++] = mate_of_v;
//           }
//           v = ws_.reuse_targets.FindNextSet(u, v);
//         }
//       }
//     }
//     return dist_null_ != std::numeric_limits<int>::max();
//   }

//   bool DFS(std::uint32_t u)
//   {
//     std::size_t v = ws_.reuse_targets.FindFirstSet(u);
//     while (v < num_tasks_)
//     {
//       std::uint32_t mate_of_v = ws_.mate_v[v];
//       if (mate_of_v == INVALID_INDEX)
//       {
//         if (dist_null_ == ws_.dist[u] + 1)
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
//       {
//         if (DFS(mate_of_v))
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       v = ws_.reuse_targets.FindNextSet(u, v);
//     }
//     ws_.dist[u] = -1;
//     return false;
//   }

//   void AssignStaticSlots()
//   {
//     task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
//     std::uint32_t next_slot_id = 0;

//     for (std::uint32_t task_id = 0; task_id < num_tasks_; ++task_id)
//     {
//       if (ws_.mate_v[task_id] == INVALID_INDEX)
//       {
//         std::uint32_t current = task_id;
//         while (current != INVALID_INDEX)
//         {
//           task_slot_ids_[current] = next_slot_id;
//           current = ws_.mate_u[current];
//         }
//         ++next_slot_id;
//       }
//     }
//   }

//   std::uint32_t num_tasks_;
//   const std::vector<Task>& task_list_;
//   const std::vector<std::uint32_t>& topo_order_;
//   std::vector<std::uint32_t>& task_slot_ids_;

//   ThreadLocalWorkspace& ws_;
//   int dist_null_ = 0;
// };

// } // namespace

// void
// CBC_SPDS::BuildTaskGraph()
// {
//   constexpr auto INCOMING = FaceOrientation::INCOMING;
//   constexpr auto OUTGOING = FaceOrientation::OUTGOING;

//   const auto num_loc_cells = grid_->local_cells.size();
//   task_list_.assign(num_loc_cells, Task{});
//   local_successor_offsets_.resize(num_loc_cells + 1, 0);

//   std::size_t successor_count = 0;
//   for (const auto& cell : grid_->local_cells)
//   {
//     unsigned int num_dependencies = 0;
//     std::vector<std::uint32_t> predecessors;
//     std::vector<std::uint32_t> successors;

//     for (std::size_t f = 0; f < cell.faces.size(); ++f)
//     {
//       const auto& face = cell.faces[f];
//       const auto orientation = cell_face_orientations_[cell.local_id][f];

//       if (orientation == INCOMING and face.has_neighbor)
//       {
//         ++num_dependencies;
//         if (face.IsNeighborLocal(grid_.get()))
//           predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
//       }
//       else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
//         successors.push_back(grid_->cells[face.neighbor_id].local_id);
//     }

//     successor_count += successors.size();
//     local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);
//     task_list_[cell.local_id] = Task{
//       0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
//   }

//   local_successors_.resize(successor_count);
//   initial_successors_to_retire_.resize(task_list_.size());
//   for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
//   {
//     initial_successors_to_retire_[cell_id] =
//       static_cast<std::uint32_t>(task_list_[cell_id].successors.size());
//     std::copy(task_list_[cell_id].successors.begin(),
//               task_list_[cell_id].successors.end(),
//               local_successors_.begin() + local_successor_offsets_[cell_id]);
//   }
// }

// CBC_SPDS::CBC_SPDS(const Vector3& omega,
//                    const std::shared_ptr<MeshContinuum>& grid,
//                    const bool allow_cycles)
//   : SPDS(omega, grid)
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

//   const auto num_loc_cells = grid->local_cells.size();

//   std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
//   std::set<int> location_successors;
//   std::set<int> location_dependencies;

//   PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

//   location_successors_.reserve(location_successors.size());
//   location_dependencies_.reserve(location_dependencies.size());

//   for (const auto loc : location_successors)
//     location_successors_.push_back(loc);

//   for (const auto loc : location_dependencies)
//     location_dependencies_.push_back(loc);

//   Graph local_dg(num_loc_cells);
//   for (std::size_t c = 0; c < num_loc_cells; ++c)
//     for (const auto& successor : cell_successors[c])
//       boost::add_edge(c, successor.first, successor.second, local_dg);

//   if (allow_cycles)
//   {
//     const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
//     for (const auto& [u, v] : edges_to_remove)
//       local_sweep_fas_.emplace_back(u, v);
//   }

//   spls_.clear();
//   boost::topological_sort(local_dg, std::back_inserter(spls_));
//   std::reverse(spls_.begin(), spls_.end());
//   if (spls_.empty())
//   {
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
//                            "Cycles need to be allowed by the calling application.");
//   }

//   topo_order_.reserve(spls_.size());
//   for (const auto v : spls_)
//     topo_order_.push_back(static_cast<std::uint32_t>(v));

//   std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
//   CommunicateLocationDependencies(location_dependencies_, global_dependencies);
//   BuildTaskGraph();

//   max_num_local_psi_slots_ = num_loc_cells;
//   task_slot_ids_.resize(num_loc_cells);
//   std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
// }

// const std::vector<Task>&
// CBC_SPDS::GetTaskList() const noexcept
// {
//   return task_list_;
// }

// void
// CBC_SPDS::ComputeMaxNumLocalPsiSlots()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

//   const std::uint32_t num_tasks = static_cast<std::uint32_t>(task_list_.size());
//   if (num_tasks == 0)
//   {
//     max_num_local_psi_slots_ = 0;
//     return;
//   }

//   // Persists exactly once per thread in the SPMD_ThreadPool
//   thread_local ThreadLocalWorkspace workspace;
  
//   DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
//   max_num_local_psi_slots_ = allocator.Solve();
// }

// #ifndef __OPENSN_WITH_GPU__
// void
// CBC_SPDS::CopyTaskGraphDataOnDevice() const
// {
// }

// void
// CBC_SPDS::FreeDeviceData() const
// {
// }
// #endif

// CBC_SPDS::~CBC_SPDS()
// {
//   FreeDeviceData();
// }

// } // namespace opensn

// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
// #include "framework/logging/log.h"
// #include "framework/mesh/mesh_continuum/mesh_continuum.h"
// #include "framework/runtime.h"
// #include "caliper/cali.h"
// #include <queue>
// #include <limits>
// #include <algorithm>
// #include <cstring>
// #include <numeric>
// #include <stdexcept>
// #include <boost/graph/topological_sort.hpp>

// #if __AVX512F__ || __AVX2__
// #include <immintrin.h>
// #endif

// namespace opensn
// {

// namespace
// {

// constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// class AVXBitMatrix
// {
// public:
// #if __AVX512F__
//   static constexpr std::size_t kWordsAlign = 8;
// #elif __AVX2__
//   static constexpr std::size_t kWordsAlign = 4;
// #else
//   static constexpr std::size_t kWordsAlign = 1;
// #endif

//   AVXBitMatrix() : n_(0), words_per_row_(0) {}

//   void ResizeAndClear(std::size_t n)
//   {
//     n_ = n;
//     words_per_row_ = ((n + 63) / 64 + kWordsAlign - 1) / kWordsAlign * kWordsAlign;
//     const std::size_t required_words = n * words_per_row_;
    
//     if (data_.size() < required_words)
//       data_.resize(required_words);
      
//     std::memset(data_.data(), 0, required_words * sizeof(std::uint64_t));
//   }

//   std::size_t WordsPerRow() const { return words_per_row_; }
//   std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }
//   const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }
  
//   void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }
//   void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

//   // Exploits topological ordering to skip the left half of the matrix
//   void CopyRow(std::size_t dst, const AVXBitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = (start_pos / 64 / kWordsAlign) * kWordsAlign;
//     const std::size_t words_to_copy = words_per_row_ - start_word;
//     std::memcpy(Row(dst) + start_word, src_mat.Row(src_row) + start_word, words_to_copy * sizeof(std::uint64_t));
//   }

//   void ClearRow(std::size_t i)
//   {
//     std::memset(Row(i), 0, words_per_row_ * sizeof(std::uint64_t));
//   }

//   // SIMD truncated loops natively handle 50% operations on Upper Triangular graphs
//   void OrRows(std::size_t dst, const AVXBitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = (start_pos / 64 / kWordsAlign) * kWordsAlign;
//     std::uint64_t* d = Row(dst) + start_word;
//     const std::uint64_t* s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_process = words_per_row_ - start_word;
//     std::size_t w = 0;

// #if __AVX512F__
//     for (; w < words_to_process; w += 8)
//     {
//       __m512i vd = _mm512_loadu_si512(reinterpret_cast<const void*>(d + w));
//       __m512i vs = _mm512_loadu_si512(reinterpret_cast<const void*>(s + w));
//       _mm512_storeu_si512(reinterpret_cast<void*>(d + w), _mm512_or_si512(vd, vs));
//     }
// #elif __AVX2__
//     for (; w < words_to_process; w += 4)
//     {
//       __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
//       __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
//       _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w), _mm256_or_si256(vd, vs));
//     }
// #else
//     for (; w < words_to_process; ++w) d[w] |= s[w];
// #endif
//   }

//   void AndRows(std::size_t dst, const AVXBitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = (start_pos / 64 / kWordsAlign) * kWordsAlign;
//     std::uint64_t* d = Row(dst) + start_word;
//     const std::uint64_t* s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_process = words_per_row_ - start_word;
//     std::size_t w = 0;

// #if __AVX512F__
//     for (; w < words_to_process; w += 8)
//     {
//       __m512i vd = _mm512_loadu_si512(reinterpret_cast<const void*>(d + w));
//       __m512i vs = _mm512_loadu_si512(reinterpret_cast<const void*>(s + w));
//       _mm512_storeu_si512(reinterpret_cast<void*>(d + w), _mm512_and_si512(vd, vs));
//     }
// #elif __AVX2__
//     for (; w < words_to_process; w += 4)
//     {
//       __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
//       __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
//       _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w), _mm256_and_si256(vd, vs));
//     }
// #else
//     for (; w < words_to_process; ++w) d[w] &= s[w];
// #endif
//   }

//   std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
//   {
//     const std::uint64_t* r = Row(row);
//     std::size_t w = start_pos / 64;

//     if (w >= words_per_row_) return n_;

//     std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));
//     if (masked)
//       return std::min(n_, w * 64 + static_cast<std::size_t>(__builtin_ctzll(masked)));

//     ++w;

// #if __AVX512F__
//     const __m512i v_zero = _mm512_setzero_si512();
//     for (; w + 7 < words_per_row_; w += 8)
//     {
//       __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(r + w));
//       if (_mm512_cmpneq_epi64_mask(v, v_zero) != 0) break;
//     }
// #elif __AVX2__
//     const __m256i v_zero = _mm256_setzero_si256();
//     for (; w + 3 < words_per_row_; w += 4)
//     {
//       __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + w));
//       if (!_mm256_testz_si256(v, v)) break;
//     }
// #endif

//     for (; w < words_per_row_; ++w)
//     {
//       if (r[w])
//         return std::min(n_, w * 64 + static_cast<std::size_t>(__builtin_ctzll(r[w])));
//     }
//     return n_;
//   }

//   std::size_t FindNextSet(std::size_t row, std::size_t pos) const 
//   { 
//     return FindFirstSet(row, pos + 1); 
//   }

// private:
//   std::size_t n_;
//   std::size_t words_per_row_;
//   std::vector<std::uint64_t> data_;
// };

// struct ThreadLocalWorkspace
// {
//   AVXBitMatrix reachability;
//   AVXBitMatrix reuse_targets;
//   std::vector<std::uint32_t> mate_u;
//   std::vector<std::uint32_t> mate_v;
//   std::vector<int> dist;
//   std::vector<std::uint32_t> queue;
//   std::vector<std::uint32_t> topo_rank;
  
//   void Prepare(std::size_t n)
//   {
//     reachability.ResizeAndClear(n);
//     reuse_targets.ResizeAndClear(n);
//     mate_u.assign(n, INVALID_INDEX);
//     mate_v.assign(n, INVALID_INDEX);
//     dist.assign(n, -1);
    
//     if (queue.size() < n) queue.resize(n);
//     if (topo_rank.size() < n) topo_rank.resize(n);
//   }
// };

// class DenseHopcroftKarp
// {
// public:
//   DenseHopcroftKarp(std::uint32_t num_tasks,
//                     const std::vector<Task>& task_list,
//                     const std::vector<std::uint32_t>& topo_order,
//                     std::vector<std::uint32_t>& task_slot_ids,
//                     ThreadLocalWorkspace& ws)
//     : num_tasks_(num_tasks),
//       task_list_(task_list),
//       topo_order_(topo_order),
//       task_slot_ids_(task_slot_ids),
//       ws_(ws)
//   {
//     ws_.Prepare(num_tasks_);
    
//     // O(1) translation from internal cell.local_id to topological rank
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//       ws_.topo_rank[topo_order_[i]] = i;
//   }

//   std::size_t Solve()
//   {
//     // 1. Build Transitive Closure Bottom-Up (Exploiting upper-triangular structure)
//     for (int i = static_cast<int>(num_tasks_) - 1; i >= 0; --i)
//     {
//       ws_.reachability.SetBit(i, i);
//       const std::uint32_t u = topo_order_[i];
//       for (const auto& succ : task_list_[u].successors)
//       {
//         std::uint32_t succ_rank = ws_.topo_rank[succ];
//         ws_.reachability.OrRows(i, ws_.reachability, succ_rank, i); 
//       }
//     }

//     // 2. Precompute Exact Reuse Intersections
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       const std::uint32_t u = topo_order_[i];
//       const auto& successors = task_list_[u].successors;
//       if (successors.empty()) continue;

//       std::uint32_t first_succ_rank = ws_.topo_rank[successors[0]];
//       ws_.reuse_targets.CopyRow(i, ws_.reachability, first_succ_rank, i);

//       for (std::size_t j = 1; j < successors.size(); ++j)
//       {
//         ws_.reuse_targets.AndRows(i, ws_.reachability, ws_.topo_rank[successors[j]], i);
//       }

//       for (const auto& succ : successors)
//         ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
//     }

//     // 3. Dense Hopcroft-Karp Bipartite Matching on Topological Space
//     std::size_t matching_size = GreedyInit();
//     while (BFS())
//     {
//       for (std::uint32_t i = 0; i < num_tasks_; ++i)
//       {
//         if (ws_.mate_u[i] == INVALID_INDEX && DFS(i))
//           ++matching_size;
//       }
//     }
    
//     AssignStaticSlots();
//     return static_cast<std::size_t>(num_tasks_) - matching_size;
//   }

// private:
//   std::size_t GreedyInit()
//   {
//     std::size_t count = 0;
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_u[i] != INVALID_INDEX) continue;
        
//       // Interval Graph Theory: Topologically earliest target generates perfect matching
//       std::size_t v = ws_.reuse_targets.FindFirstSet(i, i + 1);
//       while (v < num_tasks_)
//       {
//         if (ws_.mate_v[v] == INVALID_INDEX)
//         {
//           ws_.mate_u[i] = static_cast<std::uint32_t>(v);
//           ws_.mate_v[v] = i;
//           ++count;
//           break;
//         }
//         v = ws_.reuse_targets.FindNextSet(i, v);
//       }
//     }
//     return count;
//   }

//   bool BFS()
//   {
//     std::fill(ws_.dist.begin(), ws_.dist.begin() + num_tasks_, -1);
//     std::size_t head = 0, tail = 0;

//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_u[i] == INVALID_INDEX)
//       {
//         ws_.dist[i] = 0;
//         ws_.queue[tail++] = i;
//       }
//     }

//     dist_null_ = std::numeric_limits<int>::max();

//     while (head < tail)
//     {
//       std::uint32_t u = ws_.queue[head++];

//       if (ws_.dist[u] < dist_null_)
//       {
//         std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
//         while (v < num_tasks_)
//         {
//           std::uint32_t mate_of_v = ws_.mate_v[v];
//           if (mate_of_v == INVALID_INDEX)
//           {
//             if (dist_null_ == std::numeric_limits<int>::max())
//               dist_null_ = ws_.dist[u] + 1;
//           }
//           else if (ws_.dist[mate_of_v] == -1)
//           {
//             ws_.dist[mate_of_v] = ws_.dist[u] + 1;
//             ws_.queue[tail++] = mate_of_v;
//           }
//           v = ws_.reuse_targets.FindNextSet(u, v);
//         }
//       }
//     }
//     return dist_null_ != std::numeric_limits<int>::max();
//   }

//   bool DFS(std::uint32_t u)
//   {
//     std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
//     while (v < num_tasks_)
//     {
//       std::uint32_t mate_of_v = ws_.mate_v[v];
//       if (mate_of_v == INVALID_INDEX)
//       {
//         if (dist_null_ == ws_.dist[u] + 1)
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
//       {
//         if (DFS(mate_of_v))
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       v = ws_.reuse_targets.FindNextSet(u, v);
//     }
//     ws_.dist[u] = -1;
//     return false;
//   }

//   void AssignStaticSlots()
//   {
//     task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
//     std::uint32_t next_slot_id = 0;

//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_v[i] == INVALID_INDEX)
//       {
//         std::uint32_t current = i;
//         while (current != INVALID_INDEX)
//         {
//           // Re-translate from topological index back to application cell.local_id
//           task_slot_ids_[topo_order_[current]] = next_slot_id;
//           current = ws_.mate_u[current];
//         }
//         ++next_slot_id;
//       }
//     }
//   }

//   std::uint32_t num_tasks_;
//   const std::vector<Task>& task_list_;
//   const std::vector<std::uint32_t>& topo_order_;
//   std::vector<std::uint32_t>& task_slot_ids_;

//   ThreadLocalWorkspace& ws_;
//   int dist_null_ = 0;
// };

// } // namespace

// void
// CBC_SPDS::BuildTaskGraph()
// {
//   constexpr auto INCOMING = FaceOrientation::INCOMING;
//   constexpr auto OUTGOING = FaceOrientation::OUTGOING;

//   const auto num_loc_cells = grid_->local_cells.size();
//   task_list_.assign(num_loc_cells, Task{});
//   local_successor_offsets_.resize(num_loc_cells + 1, 0);

//   std::size_t successor_count = 0;
//   for (const auto& cell : grid_->local_cells)
//   {
//     unsigned int num_dependencies = 0;
//     std::vector<std::uint32_t> predecessors;
//     std::vector<std::uint32_t> successors;

//     for (std::size_t f = 0; f < cell.faces.size(); ++f)
//     {
//       const auto& face = cell.faces[f];
//       const auto orientation = cell_face_orientations_[cell.local_id][f];

//       if (orientation == INCOMING and face.has_neighbor)
//       {
//         ++num_dependencies;
//         if (face.IsNeighborLocal(grid_.get()))
//           predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
//       }
//       else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
//         successors.push_back(grid_->cells[face.neighbor_id].local_id);
//     }

//     successor_count += successors.size();
//     local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);
//     task_list_[cell.local_id] = Task{
//       0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
//   }

//   local_successors_.resize(successor_count);
//   initial_successors_to_retire_.resize(task_list_.size());
//   for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
//   {
//     initial_successors_to_retire_[cell_id] =
//       static_cast<std::uint32_t>(task_list_[cell_id].successors.size());
//     std::copy(task_list_[cell_id].successors.begin(),
//               task_list_[cell_id].successors.end(),
//               local_successors_.begin() + local_successor_offsets_[cell_id]);
//   }
// }

// CBC_SPDS::CBC_SPDS(const Vector3& omega,
//                    const std::shared_ptr<MeshContinuum>& grid,
//                    const bool allow_cycles)
//   : SPDS(omega, grid)
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

//   const auto num_loc_cells = grid->local_cells.size();

//   std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
//   std::set<int> location_successors;
//   std::set<int> location_dependencies;

//   PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

//   location_successors_.reserve(location_successors.size());
//   location_dependencies_.reserve(location_dependencies.size());

//   for (const auto loc : location_successors)
//     location_successors_.push_back(loc);

//   for (const auto loc : location_dependencies)
//     location_dependencies_.push_back(loc);

//   Graph local_dg(num_loc_cells);
//   for (std::size_t c = 0; c < num_loc_cells; ++c)
//     for (const auto& successor : cell_successors[c])
//       boost::add_edge(c, successor.first, successor.second, local_dg);

//   if (allow_cycles)
//   {
//     const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
//     for (const auto& [u, v] : edges_to_remove)
//       local_sweep_fas_.emplace_back(u, v);
//   }

//   spls_.clear();
//   boost::topological_sort(local_dg, std::back_inserter(spls_));
//   std::reverse(spls_.begin(), spls_.end());
//   if (spls_.empty())
//   {
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
//                            "Cycles need to be allowed by the calling application.");
//   }

//   topo_order_.reserve(spls_.size());
//   for (const auto v : spls_)
//     topo_order_.push_back(static_cast<std::uint32_t>(v));

//   std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
//   CommunicateLocationDependencies(location_dependencies_, global_dependencies);
//   BuildTaskGraph();

//   max_num_local_psi_slots_ = num_loc_cells;
//   task_slot_ids_.resize(num_loc_cells);
//   std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
// }

// const std::vector<Task>&
// CBC_SPDS::GetTaskList() const noexcept
// {
//   return task_list_;
// }

// void
// CBC_SPDS::ComputeMaxNumLocalPsiSlots()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

//   const std::uint32_t num_tasks = static_cast<std::uint32_t>(task_list_.size());
//   if (num_tasks == 0)
//   {
//     max_num_local_psi_slots_ = 0;
//     return;
//   }

//   thread_local ThreadLocalWorkspace workspace;
  
//   DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
//   max_num_local_psi_slots_ = allocator.Solve();
// }

// #ifndef __OPENSN_WITH_GPU__
// void
// CBC_SPDS::CopyTaskGraphDataOnDevice() const
// {
// }

// void
// CBC_SPDS::FreeDeviceData() const
// {
// }
// #endif

// CBC_SPDS::~CBC_SPDS()
// {
//   FreeDeviceData();
// }

// } // namespace opensn

// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
// #include "framework/logging/log.h"
// #include "framework/mesh/mesh_continuum/mesh_continuum.h"
// #include "framework/runtime.h"
// #include "caliper/cali.h"
// #include <queue>
// #include <limits>
// #include <algorithm>
// #include <cstring>
// #include <numeric>
// #include <stdexcept>
// #include <boost/graph/topological_sort.hpp>

// namespace opensn
// {

// namespace
// {

// constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// class BitMatrix
// {
// public:
//   BitMatrix() : n_(0), words_per_row_(0) {}

//   void ResizeAndClear(std::size_t n)
//   {
//     n_ = n;
//     // Pad to multiple of 8 words (512 bits) to guarantee perfect 
//     // boundary alignment for compiler auto-vectorization
//     words_per_row_ = (((n + 63) / 64) + 7) & ~7ULL;
//     const std::size_t required_words = n * words_per_row_;
    
//     if (data_.size() < required_words)
//       data_.resize(required_words);
      
//     std::memset(data_.data(), 0, required_words * sizeof(std::uint64_t));
//   }

//   std::size_t WordsPerRow() const { return words_per_row_; }
//   std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }
//   const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }
  
//   void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }
//   void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

//   // The __restrict keyword guarantees no pointer aliasing, enabling the 
//   // compiler to aggressively unroll and vectorize the loops natively.
//   void CopyRow(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = start_pos / 64;
//     std::uint64_t* __restrict d = Row(dst) + start_word;
//     const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_copy = words_per_row_ - start_word;
    
//     std::memcpy(d, s, words_to_copy * sizeof(std::uint64_t));
//   }

//   void OrRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = start_pos / 64;
//     std::uint64_t* __restrict d = Row(dst) + start_word;
//     const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_process = words_per_row_ - start_word;

//     for (std::size_t w = 0; w < words_to_process; ++w)
//       d[w] |= s[w];
//   }

//   void AndRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = start_pos / 64;
//     std::uint64_t* __restrict d = Row(dst) + start_word;
//     const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_process = words_per_row_ - start_word;

//     for (std::size_t w = 0; w < words_to_process; ++w)
//       d[w] &= s[w];
//   }

//   std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
//   {
//     const std::uint64_t* __restrict r = Row(row);
//     std::size_t w = start_pos / 64;

//     if (w >= words_per_row_) return n_;

//     std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));
//     if (masked)
//       return std::min(n_, w * 64 + static_cast<std::size_t>(__builtin_ctzll(masked)));

//     for (++w; w < words_per_row_; ++w)
//     {
//       if (r[w])
//         return std::min(n_, w * 64 + static_cast<std::size_t>(__builtin_ctzll(r[w])));
//     }
//     return n_;
//   }

//   std::size_t FindNextSet(std::size_t row, std::size_t pos) const 
//   { 
//     return FindFirstSet(row, pos + 1); 
//   }

// private:
//   std::size_t n_;
//   std::size_t words_per_row_;
//   std::vector<std::uint64_t> data_;
// };

// struct ThreadLocalWorkspace
// {
//   BitMatrix reachability;
//   BitMatrix reuse_targets;
//   std::vector<std::uint32_t> mate_u;
//   std::vector<std::uint32_t> mate_v;
//   std::vector<int> dist;
//   std::vector<std::uint32_t> queue;
//   std::vector<std::uint32_t> topo_rank;
  
//   void Prepare(std::size_t n)
//   {
//     reachability.ResizeAndClear(n);
//     reuse_targets.ResizeAndClear(n);
//     mate_u.assign(n, INVALID_INDEX);
//     mate_v.assign(n, INVALID_INDEX);
//     dist.assign(n, -1);
    
//     if (queue.size() < n) queue.resize(n);
//     if (topo_rank.size() < n) topo_rank.resize(n);
//   }
// };

// class DenseHopcroftKarp
// {
// public:
//   DenseHopcroftKarp(std::uint32_t num_tasks,
//                     const std::vector<Task>& task_list,
//                     const std::vector<std::uint32_t>& topo_order,
//                     std::vector<std::uint32_t>& task_slot_ids,
//                     ThreadLocalWorkspace& ws)
//     : num_tasks_(num_tasks),
//       task_list_(task_list),
//       topo_order_(topo_order),
//       task_slot_ids_(task_slot_ids),
//       ws_(ws)
//   {
//     ws_.Prepare(num_tasks_);
    
//     // O(1) translation from internal cell.local_id to topological rank
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//       ws_.topo_rank[topo_order_[i]] = i;
//   }

//   std::size_t Solve()
//   {
//     // 1. Build Transitive Closure Bottom-Up
//     for (int i = static_cast<int>(num_tasks_) - 1; i >= 0; --i)
//     {
//       ws_.reachability.SetBit(i, i);
//       const std::uint32_t u = topo_order_[i];
//       for (const auto& succ : task_list_[u].successors)
//       {
//         std::uint32_t succ_rank = ws_.topo_rank[succ];
//         // Mathematical Truncation: A successor can mathematically never reach
//         // a node with a topological rank lower than its own. Skip entire memory blocks.
//         ws_.reachability.OrRows(i, ws_.reachability, succ_rank, succ_rank); 
//       }
//     }

//     // 2. Precompute Exact Reuse Intersections
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       const std::uint32_t u = topo_order_[i];
//       const auto& successors = task_list_[u].successors;
//       if (successors.empty()) continue;

//       // Mathematical Truncation: The intersection of downstream targets cannot 
//       // yield a valid match prior to the highest topological rank of any immediate successor.
//       std::uint32_t max_succ_rank = ws_.topo_rank[successors[0]];
//       for (std::size_t j = 1; j < successors.size(); ++j)
//         max_succ_rank = std::max(max_succ_rank, ws_.topo_rank[successors[j]]);

//       ws_.reuse_targets.CopyRow(i, ws_.reachability, ws_.topo_rank[successors[0]], max_succ_rank);

//       for (std::size_t j = 1; j < successors.size(); ++j)
//       {
//         ws_.reuse_targets.AndRows(i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);
//       }

//       for (const auto& succ : successors)
//         ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
//     }

//     // 3. Dense Hopcroft-Karp Bipartite Matching on Topological Space
//     std::size_t matching_size = GreedyInit();
//     while (BFS())
//     {
//       for (std::uint32_t i = 0; i < num_tasks_; ++i)
//       {
//         if (ws_.mate_u[i] == INVALID_INDEX && DFS(i))
//           ++matching_size;
//       }
//     }
    
//     AssignStaticSlots();
//     return static_cast<std::size_t>(num_tasks_) - matching_size;
//   }

// private:
//   std::size_t GreedyInit()
//   {
//     std::size_t count = 0;
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_u[i] != INVALID_INDEX) continue;
        
//       // Interval Graph Theory: Topologically earliest target generates perfect/near-perfect matching
//       std::size_t v = ws_.reuse_targets.FindFirstSet(i, i + 1);
//       while (v < num_tasks_)
//       {
//         if (ws_.mate_v[v] == INVALID_INDEX)
//         {
//           ws_.mate_u[i] = static_cast<std::uint32_t>(v);
//           ws_.mate_v[v] = i;
//           ++count;
//           break;
//         }
//         v = ws_.reuse_targets.FindNextSet(i, v);
//       }
//     }
//     return count;
//   }

//   bool BFS()
//   {
//     std::fill(ws_.dist.begin(), ws_.dist.begin() + num_tasks_, -1);
//     std::size_t head = 0, tail = 0;

//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_u[i] == INVALID_INDEX)
//       {
//         ws_.dist[i] = 0;
//         ws_.queue[tail++] = i;
//       }
//     }

//     dist_null_ = std::numeric_limits<int>::max();

//     while (head < tail)
//     {
//       std::uint32_t u = ws_.queue[head++];

//       if (ws_.dist[u] < dist_null_)
//       {
//         std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
//         while (v < num_tasks_)
//         {
//           std::uint32_t mate_of_v = ws_.mate_v[v];
//           if (mate_of_v == INVALID_INDEX)
//           {
//             if (dist_null_ == std::numeric_limits<int>::max())
//               dist_null_ = ws_.dist[u] + 1;
//           }
//           else if (ws_.dist[mate_of_v] == -1)
//           {
//             ws_.dist[mate_of_v] = ws_.dist[u] + 1;
//             ws_.queue[tail++] = mate_of_v;
//           }
//           v = ws_.reuse_targets.FindNextSet(u, v);
//         }
//       }
//     }
//     return dist_null_ != std::numeric_limits<int>::max();
//   }

//   bool DFS(std::uint32_t u)
//   {
//     std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
//     while (v < num_tasks_)
//     {
//       std::uint32_t mate_of_v = ws_.mate_v[v];
//       if (mate_of_v == INVALID_INDEX)
//       {
//         if (dist_null_ == ws_.dist[u] + 1)
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
//       {
//         if (DFS(mate_of_v))
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       v = ws_.reuse_targets.FindNextSet(u, v);
//     }
//     ws_.dist[u] = -1;
//     return false;
//   }

//   void AssignStaticSlots()
//   {
//     task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
//     std::uint32_t next_slot_id = 0;

//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_v[i] == INVALID_INDEX)
//       {
//         std::uint32_t current = i;
//         while (current != INVALID_INDEX)
//         {
//           // Re-translate from topological index back to application cell.local_id
//           task_slot_ids_[topo_order_[current]] = next_slot_id;
//           current = ws_.mate_u[current];
//         }
//         ++next_slot_id;
//       }
//     }
//   }

//   std::uint32_t num_tasks_;
//   const std::vector<Task>& task_list_;
//   const std::vector<std::uint32_t>& topo_order_;
//   std::vector<std::uint32_t>& task_slot_ids_;

//   ThreadLocalWorkspace& ws_;
//   int dist_null_ = 0;
// };

// } // namespace

// void
// CBC_SPDS::BuildTaskGraph()
// {
//   constexpr auto INCOMING = FaceOrientation::INCOMING;
//   constexpr auto OUTGOING = FaceOrientation::OUTGOING;

//   const auto num_loc_cells = grid_->local_cells.size();
//   task_list_.assign(num_loc_cells, Task{});
//   local_successor_offsets_.resize(num_loc_cells + 1, 0);

//   std::size_t successor_count = 0;
//   for (const auto& cell : grid_->local_cells)
//   {
//     unsigned int num_dependencies = 0;
//     std::vector<std::uint32_t> predecessors;
//     std::vector<std::uint32_t> successors;

//     for (std::size_t f = 0; f < cell.faces.size(); ++f)
//     {
//       const auto& face = cell.faces[f];
//       const auto orientation = cell_face_orientations_[cell.local_id][f];

//       if (orientation == INCOMING and face.has_neighbor)
//       {
//         ++num_dependencies;
//         if (face.IsNeighborLocal(grid_.get()))
//           predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
//       }
//       else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
//         successors.push_back(grid_->cells[face.neighbor_id].local_id);
//     }

//     successor_count += successors.size();
//     local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);
//     task_list_[cell.local_id] = Task{
//       0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
//   }

//   local_successors_.resize(successor_count);
//   initial_successors_to_retire_.resize(task_list_.size());
//   for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
//   {
//     initial_successors_to_retire_[cell_id] =
//       static_cast<std::uint32_t>(task_list_[cell_id].successors.size());
//     std::copy(task_list_[cell_id].successors.begin(),
//               task_list_[cell_id].successors.end(),
//               local_successors_.begin() + local_successor_offsets_[cell_id]);
//   }
// }

// CBC_SPDS::CBC_SPDS(const Vector3& omega,
//                    const std::shared_ptr<MeshContinuum>& grid,
//                    const bool allow_cycles)
//   : SPDS(omega, grid)
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

//   const auto num_loc_cells = grid->local_cells.size();

//   std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
//   std::set<int> location_successors;
//   std::set<int> location_dependencies;

//   PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

//   location_successors_.reserve(location_successors.size());
//   location_dependencies_.reserve(location_dependencies.size());

//   for (const auto loc : location_successors)
//     location_successors_.push_back(loc);

//   for (const auto loc : location_dependencies)
//     location_dependencies_.push_back(loc);

//   Graph local_dg(num_loc_cells);
//   for (std::size_t c = 0; c < num_loc_cells; ++c)
//     for (const auto& successor : cell_successors[c])
//       boost::add_edge(c, successor.first, successor.second, local_dg);

//   if (allow_cycles)
//   {
//     const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
//     for (const auto& [u, v] : edges_to_remove)
//       local_sweep_fas_.emplace_back(u, v);
//   }

//   spls_.clear();
//   boost::topological_sort(local_dg, std::back_inserter(spls_));
//   std::reverse(spls_.begin(), spls_.end());
//   if (spls_.empty())
//   {
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
//                            "Cycles need to be allowed by the calling application.");
//   }

//   topo_order_.reserve(spls_.size());
//   for (const auto v : spls_)
//     topo_order_.push_back(static_cast<std::uint32_t>(v));

//   std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
//   CommunicateLocationDependencies(location_dependencies_, global_dependencies);
//   BuildTaskGraph();

//   max_num_local_psi_slots_ = num_loc_cells;
//   task_slot_ids_.resize(num_loc_cells);
//   std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
// }

// const std::vector<Task>&
// CBC_SPDS::GetTaskList() const noexcept
// {
//   return task_list_;
// }

// void
// CBC_SPDS::ComputeMaxNumLocalPsiSlots()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

//   const std::uint32_t num_tasks = static_cast<std::uint32_t>(task_list_.size());
//   if (num_tasks == 0)
//   {
//     max_num_local_psi_slots_ = 0;
//     return;
//   }

//   // Persists exactly once per thread in the SPMD_ThreadPool
//   thread_local ThreadLocalWorkspace workspace;
  
//   DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
//   max_num_local_psi_slots_ = allocator.Solve();
// }

// #ifndef __OPENSN_WITH_GPU__
// void
// CBC_SPDS::CopyTaskGraphDataOnDevice() const
// {
// }

// void
// CBC_SPDS::FreeDeviceData() const
// {
// }
// #endif

// CBC_SPDS::~CBC_SPDS()
// {
//   FreeDeviceData();
// }

// } // namespace opensn

// This version should be the one that I stick with
// // SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// // SPDX-License-Identifier: MIT

// #include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
// #include "framework/logging/log.h"
// #include "framework/mesh/mesh_continuum/mesh_continuum.h"
// #include "framework/runtime.h"
// #include "caliper/cali.h"
// #include <queue>
// #include <limits>
// #include <algorithm>
// #include <cstring>
// #include <numeric>
// #include <stdexcept>
// #include <bit> // C++20 hardware bit-manipulation
// #include <boost/graph/topological_sort.hpp>

// namespace opensn
// {

// namespace
// {

// constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// class BitMatrix
// {
// public:
//   BitMatrix() : n_(0), words_per_row_(0) {}

//   void ResizeAndClear(std::size_t n)
//   {
//     n_ = n;
//     // Pad to multiple of 8 words (512 bits) to guarantee perfect 
//     // boundary alignment for compiler auto-vectorization
//     words_per_row_ = (((n + 63) / 64) + 7) & ~7ULL;
//     const std::size_t required_words = n * words_per_row_;
    
//     // Elide double-initialization overhead during growth
//     if (data_.size() < required_words)
//       data_.resize(required_words, 0ULL);
//     else
//       std::fill_n(data_.begin(), required_words, 0ULL);
//   }

//   std::size_t WordsPerRow() const { return words_per_row_; }
//   std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }
//   const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }
  
//   void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }
//   void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

//   // The __restrict keyword guarantees no pointer aliasing, enabling the 
//   // compiler to aggressively unroll and vectorize the loops natively.
//   void CopyRow(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = start_pos / 64;
//     std::uint64_t* __restrict d = Row(dst) + start_word;
//     const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_copy = words_per_row_ - start_word;
    
//     std::memcpy(d, s, words_to_copy * sizeof(std::uint64_t));
//   }

//   void OrRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = start_pos / 64;
//     std::uint64_t* __restrict d = Row(dst) + start_word;
//     const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_process = words_per_row_ - start_word;

//     for (std::size_t w = 0; w < words_to_process; ++w)
//       d[w] |= s[w];
//   }

//   void AndRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
//   {
//     const std::size_t start_word = start_pos / 64;
//     std::uint64_t* __restrict d = Row(dst) + start_word;
//     const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
//     const std::size_t words_to_process = words_per_row_ - start_word;

//     for (std::size_t w = 0; w < words_to_process; ++w)
//       d[w] &= s[w];
//   }

//   std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
//   {
//     const std::uint64_t* __restrict r = Row(row);
//     std::size_t w = start_pos / 64;

//     if (w >= words_per_row_) return n_;

//     std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));
    
//     // Mathematical Truncation: Padding bits are strictly guaranteed to be 0.
//     // Branchless return eliminates boundary checks. std::countr_zero maps to hardware TZCNT.
//     if (masked)
//       return w * 64 + static_cast<std::size_t>(std::countr_zero(masked));

//     for (++w; w < words_per_row_; ++w)
//     {
//       if (r[w])
//         return w * 64 + static_cast<std::size_t>(std::countr_zero(r[w]));
//     }
//     return n_;
//   }

//   std::size_t FindNextSet(std::size_t row, std::size_t pos) const 
//   { 
//     return FindFirstSet(row, pos + 1); 
//   }

// private:
//   std::size_t n_;
//   std::size_t words_per_row_;
//   std::vector<std::uint64_t> data_;
// };

// struct ThreadLocalWorkspace
// {
//   BitMatrix reachability;
//   BitMatrix reuse_targets;
//   std::vector<std::uint32_t> mate_u;
//   std::vector<std::uint32_t> mate_v;
//   std::vector<int> dist;
//   std::vector<std::uint32_t> queue;
//   std::vector<std::uint32_t> topo_rank;
//   std::vector<std::uint32_t> dfs_ptr; // Dead-edge tracker for bounded O(E) DFS
  
//   void Prepare(std::size_t n)
//   {
//     reachability.ResizeAndClear(n);
//     reuse_targets.ResizeAndClear(n);
//     mate_u.assign(n, INVALID_INDEX);
//     mate_v.assign(n, INVALID_INDEX);
//     dist.assign(n, -1);
    
//     if (queue.size() < n) queue.resize(n);
//     if (topo_rank.size() < n) topo_rank.resize(n);
//     if (dfs_ptr.size() < n) dfs_ptr.resize(n);
//   }
// };

// class DenseHopcroftKarp
// {
// public:
//   DenseHopcroftKarp(std::uint32_t num_tasks,
//                     const std::vector<Task>& task_list,
//                     const std::vector<std::uint32_t>& topo_order,
//                     std::vector<std::uint32_t>& task_slot_ids,
//                     ThreadLocalWorkspace& ws)
//     : num_tasks_(num_tasks),
//       task_list_(task_list),
//       topo_order_(topo_order),
//       task_slot_ids_(task_slot_ids),
//       ws_(ws)
//   {
//     ws_.Prepare(num_tasks_);
    
//     // O(1) translation from internal cell.local_id to topological rank
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//       ws_.topo_rank[topo_order_[i]] = i;
//   }

//   std::size_t Solve()
//   {
//     // 1. Build Transitive Closure Bottom-Up
//     for (std::uint32_t i = num_tasks_; i-- > 0; )
//     {
//       const std::uint32_t u = topo_order_[i];
//       const auto& successors = task_list_[u].successors;

//       if (successors.empty())
//       {
//         ws_.reachability.SetBit(i, i);
//       }
//       else
//       {
//         // Copy-Init Elision: Direct memory copy from the first successor bypasses 
//         // a fully redundant bitwise OR over an empty initialization row.
//         std::uint32_t first_succ_rank = ws_.topo_rank[successors[0]];
//         ws_.reachability.CopyRow(i, ws_.reachability, first_succ_rank, first_succ_rank);
//         ws_.reachability.SetBit(i, i);

//         for (std::size_t j = 1; j < successors.size(); ++j)
//         {
//           std::uint32_t succ_rank = ws_.topo_rank[successors[j]];
//           ws_.reachability.OrRows(i, ws_.reachability, succ_rank, succ_rank); 
//         }
//       }
//     }

//     // 2. Precompute Exact Reuse Intersections
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       const std::uint32_t u = topo_order_[i];
//       const auto& successors = task_list_[u].successors;
//       if (successors.empty()) continue;

//       // Mathematical Truncation: The intersection of downstream targets cannot 
//       // yield a valid match prior to the highest topological rank of any immediate successor.
//       std::uint32_t max_succ_rank = ws_.topo_rank[successors[0]];
//       for (std::size_t j = 1; j < successors.size(); ++j)
//         max_succ_rank = std::max(max_succ_rank, ws_.topo_rank[successors[j]]);

//       ws_.reuse_targets.CopyRow(i, ws_.reachability, ws_.topo_rank[successors[0]], max_succ_rank);

//       for (std::size_t j = 1; j < successors.size(); ++j)
//         ws_.reuse_targets.AndRows(i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);

//       for (const auto& succ : successors)
//         ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
//     }

//     // 3. Dense Hopcroft-Karp Bipartite Matching
//     std::size_t matching_size = GreedyInit();
//     while (BFS())
//     {
//       // Reset dead-edge trackers at the start of a new phase.
//       // Because reuse targets are strictly downstream, scanning starts at i + 1.
//       for (std::uint32_t i = 0; i < num_tasks_; ++i)
//         ws_.dfs_ptr[i] = i + 1;

//       for (std::uint32_t i = 0; i < num_tasks_; ++i)
//       {
//         if (ws_.mate_u[i] == INVALID_INDEX && DFS(i))
//           ++matching_size;
//       }
//     }
    
//     AssignStaticSlots();
//     return static_cast<std::size_t>(num_tasks_) - matching_size;
//   }

// private:
//   std::size_t GreedyInit()
//   {
//     std::size_t count = 0;
//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_u[i] != INVALID_INDEX) continue;
        
//       std::size_t v = ws_.reuse_targets.FindFirstSet(i, i + 1);
//       while (v < num_tasks_)
//       {
//         if (ws_.mate_v[v] == INVALID_INDEX)
//         {
//           ws_.mate_u[i] = static_cast<std::uint32_t>(v);
//           ws_.mate_v[v] = i;
//           ++count;
//           break;
//         }
//         v = ws_.reuse_targets.FindNextSet(i, v);
//       }
//     }
//     return count;
//   }

//   bool BFS()
//   {
//     std::fill_n(ws_.dist.begin(), num_tasks_, -1);
//     std::size_t head = 0, tail = 0;

//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_u[i] == INVALID_INDEX)
//       {
//         ws_.dist[i] = 0;
//         ws_.queue[tail++] = i;
//       }
//     }

//     dist_null_ = std::numeric_limits<int>::max();

//     while (head < tail)
//     {
//       std::uint32_t u = ws_.queue[head++];

//       if (ws_.dist[u] < dist_null_)
//       {
//         std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
//         while (v < num_tasks_)
//         {
//           std::uint32_t mate_of_v = ws_.mate_v[v];
//           if (mate_of_v == INVALID_INDEX)
//           {
//             if (dist_null_ == std::numeric_limits<int>::max())
//               dist_null_ = ws_.dist[u] + 1;
//           }
//           else if (ws_.dist[mate_of_v] == -1)
//           {
//             ws_.dist[mate_of_v] = ws_.dist[u] + 1;
//             ws_.queue[tail++] = mate_of_v;
//           }
//           v = ws_.reuse_targets.FindNextSet(u, v);
//         }
//       }
//     }
//     return dist_null_ != std::numeric_limits<int>::max();
//   }

//   bool DFS(std::uint32_t u)
//   {
//     // Resume DFS from the last unevaluated edge in this phase
//     for (std::size_t v = ws_.reuse_targets.FindFirstSet(u, ws_.dfs_ptr[u]); 
//          v < num_tasks_; 
//          v = ws_.reuse_targets.FindFirstSet(u, ws_.dfs_ptr[u]))
//     {
//       ws_.dfs_ptr[u] = v + 1; // Dead-edge tracking guarantees O(E) evaluation per phase
      
//       std::uint32_t mate_of_v = ws_.mate_v[v];
//       if (mate_of_v == INVALID_INDEX)
//       {
//         if (dist_null_ == ws_.dist[u] + 1)
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//       else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
//       {
//         if (DFS(mate_of_v))
//         {
//           ws_.mate_v[v] = u;
//           ws_.mate_u[u] = static_cast<std::uint32_t>(v);
//           ws_.dist[u] = -1;
//           return true;
//         }
//       }
//     }
//     ws_.dist[u] = -1;
//     return false;
//   }

//   void AssignStaticSlots()
//   {
//     task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
//     std::uint32_t next_slot_id = 0;

//     for (std::uint32_t i = 0; i < num_tasks_; ++i)
//     {
//       if (ws_.mate_v[i] == INVALID_INDEX)
//       {
//         std::uint32_t current = i;
//         while (current != INVALID_INDEX)
//         {
//           task_slot_ids_[topo_order_[current]] = next_slot_id;
//           current = ws_.mate_u[current];
//         }
//         ++next_slot_id;
//       }
//     }
//   }

//   std::uint32_t num_tasks_;
//   const std::vector<Task>& task_list_;
//   const std::vector<std::uint32_t>& topo_order_;
//   std::vector<std::uint32_t>& task_slot_ids_;

//   ThreadLocalWorkspace& ws_;
//   int dist_null_ = 0;
// };

// } // namespace

// void
// CBC_SPDS::BuildTaskGraph()
// {
//   constexpr auto INCOMING = FaceOrientation::INCOMING;
//   constexpr auto OUTGOING = FaceOrientation::OUTGOING;

//   const auto num_loc_cells = grid_->local_cells.size();
//   task_list_.assign(num_loc_cells, Task{});
//   local_successor_offsets_.resize(num_loc_cells + 1, 0);

//   std::size_t successor_count = 0;
//   for (const auto& cell : grid_->local_cells)
//   {
//     unsigned int num_dependencies = 0;
//     std::vector<std::uint32_t> predecessors;
//     std::vector<std::uint32_t> successors;

//     // Pre-allocate to prevent vector expansion during face checks
//     predecessors.reserve(cell.faces.size());
//     successors.reserve(cell.faces.size());

//     for (std::size_t f = 0; f < cell.faces.size(); ++f)
//     {
//       const auto& face = cell.faces[f];
//       const auto orientation = cell_face_orientations_[cell.local_id][f];

//       if (orientation == INCOMING and face.has_neighbor)
//       {
//         ++num_dependencies;
//         if (face.IsNeighborLocal(grid_.get()))
//           predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
//       }
//       else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
//         successors.push_back(grid_->cells[face.neighbor_id].local_id);
//     }

//     successor_count += successors.size();
//     local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);
//     task_list_[cell.local_id] = Task{
//       0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
//   }

//   local_successors_.resize(successor_count);
//   initial_successors_to_retire_.resize(task_list_.size());
//   for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
//   {
//     initial_successors_to_retire_[cell_id] =
//       static_cast<std::uint32_t>(task_list_[cell_id].successors.size());
//     std::copy(task_list_[cell_id].successors.begin(),
//               task_list_[cell_id].successors.end(),
//               local_successors_.begin() + local_successor_offsets_[cell_id]);
//   }
// }

// CBC_SPDS::CBC_SPDS(const Vector3& omega,
//                    const std::shared_ptr<MeshContinuum>& grid,
//                    const bool allow_cycles)
//   : SPDS(omega, grid)
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

//   const auto num_loc_cells = grid->local_cells.size();

//   std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
//   std::set<int> location_successors;
//   std::set<int> location_dependencies;

//   PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

//   location_successors_.reserve(location_successors.size());
//   location_dependencies_.reserve(location_dependencies.size());

//   for (const auto loc : location_successors)
//     location_successors_.push_back(loc);

//   for (const auto loc : location_dependencies)
//     location_dependencies_.push_back(loc);

//   Graph local_dg(num_loc_cells);
//   for (std::size_t c = 0; c < num_loc_cells; ++c)
//     for (const auto& successor : cell_successors[c])
//       boost::add_edge(c, successor.first, successor.second, local_dg);

//   if (allow_cycles)
//   {
//     const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
//     for (const auto& [u, v] : edges_to_remove)
//       local_sweep_fas_.emplace_back(u, v);
//   }

//   spls_.clear();
//   boost::topological_sort(local_dg, std::back_inserter(spls_));
//   std::reverse(spls_.begin(), spls_.end());
//   if (spls_.empty())
//   {
//     throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
//                            "Cycles need to be allowed by the calling application.");
//   }

//   topo_order_.reserve(spls_.size());
//   for (const auto v : spls_)
//     topo_order_.push_back(static_cast<std::uint32_t>(v));

//   std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
//   CommunicateLocationDependencies(location_dependencies_, global_dependencies);
//   BuildTaskGraph();

//   max_num_local_psi_slots_ = num_loc_cells;
//   task_slot_ids_.resize(num_loc_cells);
//   std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
// }

// const std::vector<Task>&
// CBC_SPDS::GetTaskList() const noexcept
// {
//   return task_list_;
// }

// void
// CBC_SPDS::ComputeMaxNumLocalPsiSlots()
// {
//   CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

//   const std::uint32_t num_tasks = static_cast<std::uint32_t>(task_list_.size());
//   if (num_tasks == 0)
//   {
//     max_num_local_psi_slots_ = 0;
//     return;
//   }

//   thread_local ThreadLocalWorkspace workspace;
  
//   DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
//   max_num_local_psi_slots_ = allocator.Solve();
// }

// #ifndef __OPENSN_WITH_GPU__
// void
// CBC_SPDS::CopyTaskGraphDataOnDevice() const
// {
// }

// void
// CBC_SPDS::FreeDeviceData() const
// {
// }
// #endif

// CBC_SPDS::~CBC_SPDS()
// {
//   FreeDeviceData();
// }

// } // namespace opensn

// No, this is the version I should go with as it removes the dfs_ptr entirely
// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "framework/logging/log.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <queue>
#include <limits>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <bit>
#include <boost/graph/topological_sort.hpp>

namespace opensn
{

namespace
{

constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

class BitMatrix
{
public:
  BitMatrix() : n_(0), words_per_row_(0) {}

  void ResizeAndClear(std::size_t n)
  {
    n_ = n;
    words_per_row_ = (((n + 63) / 64) + 7) & ~7ULL;
    const std::size_t required_words = n * words_per_row_;
    
    if (data_.size() < required_words)
      data_.resize(required_words, 0ULL);
    else
      std::fill_n(data_.begin(), required_words, 0ULL);
  }

  std::size_t WordsPerRow() const { return words_per_row_; }
  std::uint64_t* Row(std::size_t i) { return data_.data() + i * words_per_row_; }
  const std::uint64_t* Row(std::size_t i) const { return data_.data() + i * words_per_row_; }
  
  void SetBit(std::size_t i, std::size_t j) { Row(i)[j / 64] |= (1ULL << (j % 64)); }
  void ClearBit(std::size_t i, std::size_t j) { Row(i)[j / 64] &= ~(1ULL << (j % 64)); }

  void CopyRow(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_copy = words_per_row_ - start_word;
    
    std::memcpy(d, s, words_to_copy * sizeof(std::uint64_t));
  }

  void OrRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] |= s[w];
  }

  void AndRows(std::size_t dst, const BitMatrix& src_mat, std::size_t src_row, std::size_t start_pos = 0)
  {
    const std::size_t start_word = start_pos / 64;
    std::uint64_t* __restrict d = Row(dst) + start_word;
    const std::uint64_t* __restrict s = src_mat.Row(src_row) + start_word;
    const std::size_t words_to_process = words_per_row_ - start_word;

    for (std::size_t w = 0; w < words_to_process; ++w)
      d[w] &= s[w];
  }

  std::size_t FindFirstSet(std::size_t row, std::size_t start_pos = 0) const
  {
    const std::uint64_t* __restrict r = Row(row);
    std::size_t w = start_pos / 64;

    if (w >= words_per_row_) return n_;

    std::uint64_t masked = r[w] & (~0ULL << (start_pos % 64));
    
    if (masked)
      return w * 64 + static_cast<std::size_t>(std::countr_zero(masked));

    for (++w; w < words_per_row_; ++w)
    {
      if (r[w])
        return w * 64 + static_cast<std::size_t>(std::countr_zero(r[w]));
    }
    return n_;
  }

  std::size_t FindNextSet(std::size_t row, std::size_t pos) const 
  { 
    return FindFirstSet(row, pos + 1); 
  }

private:
  std::size_t n_;
  std::size_t words_per_row_;
  std::vector<std::uint64_t> data_;
};

struct ThreadLocalWorkspace
{
  BitMatrix reachability;
  BitMatrix reuse_targets;
  std::vector<std::uint32_t> mate_u;
  std::vector<std::uint32_t> mate_v;
  std::vector<int> dist;
  std::vector<std::uint32_t> queue;
  std::vector<std::uint32_t> topo_rank;
  
  void Prepare(std::size_t n)
  {
    reachability.ResizeAndClear(n);
    reuse_targets.ResizeAndClear(n);
    mate_u.assign(n, INVALID_INDEX);
    mate_v.assign(n, INVALID_INDEX);
    dist.assign(n, -1);
    
    if (queue.size() < n) queue.resize(n);
    if (topo_rank.size() < n) topo_rank.resize(n);
  }
};

class DenseHopcroftKarp
{
public:
  DenseHopcroftKarp(std::uint32_t num_tasks,
                    const std::vector<Task>& task_list,
                    const std::vector<std::uint32_t>& topo_order,
                    std::vector<std::uint32_t>& task_slot_ids,
                    ThreadLocalWorkspace& ws)
    : num_tasks_(num_tasks),
      task_list_(task_list),
      topo_order_(topo_order),
      task_slot_ids_(task_slot_ids),
      ws_(ws)
  {
    ws_.Prepare(num_tasks_);
    
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
      ws_.topo_rank[topo_order_[i]] = i;
  }

  std::size_t Solve()
  {
    for (std::uint32_t i = num_tasks_; i-- > 0; )
    {
      const std::uint32_t u = topo_order_[i];
      const auto& successors = task_list_[u].successors;

      if (successors.empty())
      {
        ws_.reachability.SetBit(i, i);
      }
      else
      {
        std::uint32_t first_succ_rank = ws_.topo_rank[successors[0]];
        ws_.reachability.CopyRow(i, ws_.reachability, first_succ_rank, first_succ_rank);
        ws_.reachability.SetBit(i, i);

        for (std::size_t j = 1; j < successors.size(); ++j)
        {
          std::uint32_t succ_rank = ws_.topo_rank[successors[j]];
          ws_.reachability.OrRows(i, ws_.reachability, succ_rank, succ_rank); 
        }
      }
    }

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      const std::uint32_t u = topo_order_[i];
      const auto& successors = task_list_[u].successors;
      if (successors.empty()) continue;

      std::uint32_t max_succ_rank = ws_.topo_rank[successors[0]];
      for (std::size_t j = 1; j < successors.size(); ++j)
        max_succ_rank = std::max(max_succ_rank, ws_.topo_rank[successors[j]]);

      ws_.reuse_targets.CopyRow(i, ws_.reachability, ws_.topo_rank[successors[0]], max_succ_rank);

      for (std::size_t j = 1; j < successors.size(); ++j)
        ws_.reuse_targets.AndRows(i, ws_.reachability, ws_.topo_rank[successors[j]], max_succ_rank);

      for (const auto& succ : successors)
        ws_.reuse_targets.ClearBit(i, ws_.topo_rank[succ]);
    }

    std::size_t matching_size = GreedyInit();
    while (BFS())
    {
      for (std::uint32_t i = 0; i < num_tasks_; ++i)
      {
        if (ws_.mate_u[i] == INVALID_INDEX && DFS(i))
          ++matching_size;
      }
    }
    
    AssignStaticSlots();
    return static_cast<std::size_t>(num_tasks_) - matching_size;
  }

private:
  std::size_t GreedyInit()
  {
    std::size_t count = 0;
    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] != INVALID_INDEX) continue;
        
      std::size_t v = ws_.reuse_targets.FindFirstSet(i, i + 1);
      while (v < num_tasks_)
      {
        if (ws_.mate_v[v] == INVALID_INDEX)
        {
          ws_.mate_u[i] = static_cast<std::uint32_t>(v);
          ws_.mate_v[v] = i;
          ++count;
          break;
        }
        v = ws_.reuse_targets.FindNextSet(i, v);
      }
    }
    return count;
  }

  bool BFS()
  {
    std::fill_n(ws_.dist.begin(), num_tasks_, -1);
    std::size_t head = 0, tail = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_u[i] == INVALID_INDEX)
      {
        ws_.dist[i] = 0;
        ws_.queue[tail++] = i;
      }
    }

    dist_null_ = std::numeric_limits<int>::max();

    while (head < tail)
    {
      std::uint32_t u = ws_.queue[head++];

      if (ws_.dist[u] < dist_null_)
      {
        std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
        while (v < num_tasks_)
        {
          std::uint32_t mate_of_v = ws_.mate_v[v];
          if (mate_of_v == INVALID_INDEX)
          {
            if (dist_null_ == std::numeric_limits<int>::max())
              dist_null_ = ws_.dist[u] + 1;
          }
          else if (ws_.dist[mate_of_v] == -1)
          {
            ws_.dist[mate_of_v] = ws_.dist[u] + 1;
            ws_.queue[tail++] = mate_of_v;
          }
          v = ws_.reuse_targets.FindNextSet(u, v);
        }
      }
    }
    return dist_null_ != std::numeric_limits<int>::max();
  }

  bool DFS(std::uint32_t u)
  {
    std::size_t v = ws_.reuse_targets.FindFirstSet(u, u + 1);
    while (v < num_tasks_)
    {
      std::uint32_t mate_of_v = ws_.mate_v[v];
      if (mate_of_v == INVALID_INDEX)
      {
        if (dist_null_ == ws_.dist[u] + 1)
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
      else if (ws_.dist[mate_of_v] == ws_.dist[u] + 1)
      {
        if (DFS(mate_of_v))
        {
          ws_.mate_v[v] = u;
          ws_.mate_u[u] = static_cast<std::uint32_t>(v);
          ws_.dist[u] = -1;
          return true;
        }
      }
      v = ws_.reuse_targets.FindNextSet(u, v);
    }
    ws_.dist[u] = -1;
    return false;
  }

  void AssignStaticSlots()
  {
    task_slot_ids_.assign(num_tasks_, INVALID_INDEX);
    std::uint32_t next_slot_id = 0;

    for (std::uint32_t i = 0; i < num_tasks_; ++i)
    {
      if (ws_.mate_v[i] == INVALID_INDEX)
      {
        std::uint32_t current = i;
        while (current != INVALID_INDEX)
        {
          task_slot_ids_[topo_order_[current]] = next_slot_id;
          current = ws_.mate_u[current];
        }
        ++next_slot_id;
      }
    }
  }

  std::uint32_t num_tasks_;
  const std::vector<Task>& task_list_;
  const std::vector<std::uint32_t>& topo_order_;
  std::vector<std::uint32_t>& task_slot_ids_;

  ThreadLocalWorkspace& ws_;
  int dist_null_ = 0;
};

} // namespace

void
CBC_SPDS::BuildTaskGraph()
{
  constexpr auto INCOMING = FaceOrientation::INCOMING;
  constexpr auto OUTGOING = FaceOrientation::OUTGOING;

  const auto num_loc_cells = grid_->local_cells.size();
  task_list_.assign(num_loc_cells, Task{});
  local_successor_offsets_.resize(num_loc_cells + 1, 0);

  std::size_t successor_count = 0;
  for (const auto& cell : grid_->local_cells)
  {
    unsigned int num_dependencies = 0;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> successors;

    predecessors.reserve(cell.faces.size());
    successors.reserve(cell.faces.size());

    for (std::size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      const auto orientation = cell_face_orientations_[cell.local_id][f];

      if (orientation == INCOMING and face.has_neighbor)
      {
        ++num_dependencies;
        if (face.IsNeighborLocal(grid_.get()))
          predecessors.push_back(grid_->cells[face.neighbor_id].local_id);
      }
      else if (orientation == OUTGOING and face.has_neighbor and face.IsNeighborLocal(grid_.get()))
        successors.push_back(grid_->cells[face.neighbor_id].local_id);
    }

    successor_count += successors.size();
    local_successor_offsets_[cell.local_id + 1] = static_cast<std::uint32_t>(successor_count);
    task_list_[cell.local_id] = Task{
      0, num_dependencies, std::move(predecessors), std::move(successors), cell.local_id, &cell};
  }

  local_successors_.resize(successor_count);
  initial_successors_to_retire_.resize(task_list_.size());
  for (std::uint32_t cell_id = 0; cell_id < task_list_.size(); ++cell_id)
  {
    initial_successors_to_retire_[cell_id] =
      static_cast<std::uint32_t>(task_list_[cell_id].successors.size());
    std::copy(task_list_[cell_id].successors.begin(),
              task_list_[cell_id].successors.end(),
              local_successors_.begin() + local_successor_offsets_[cell_id]);
  }
}

CBC_SPDS::CBC_SPDS(const Vector3& omega,
                   const std::shared_ptr<MeshContinuum>& grid,
                   const bool allow_cycles)
  : SPDS(omega, grid)
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::CBC_SPDS");

  const auto num_loc_cells = grid->local_cells.size();

  std::vector<std::set<std::pair<std::uint32_t, double>>> cell_successors(num_loc_cells);
  std::set<int> location_successors;
  std::set<int> location_dependencies;

  PopulateCellRelationships(omega, location_dependencies, location_successors, cell_successors);

  location_successors_.reserve(location_successors.size());
  location_dependencies_.reserve(location_dependencies.size());

  for (const auto loc : location_successors)
    location_successors_.push_back(loc);

  for (const auto loc : location_dependencies)
    location_dependencies_.push_back(loc);

  Graph local_dg(num_loc_cells);
  for (std::size_t c = 0; c < num_loc_cells; ++c)
    for (const auto& successor : cell_successors[c])
      boost::add_edge(c, successor.first, successor.second, local_dg);

  if (allow_cycles)
  {
    const auto edges_to_remove = RemoveCyclicDependencies(local_dg);
    for (const auto& [u, v] : edges_to_remove)
      local_sweep_fas_.emplace_back(u, v);
  }

  spls_.clear();
  boost::topological_sort(local_dg, std::back_inserter(spls_));
  std::reverse(spls_.begin(), spls_.end());
  if (spls_.empty())
  {
    throw std::logic_error("CBC_SPDS: Cyclic dependencies found in the local cell graph.\n"
                           "Cycles need to be allowed by the calling application.");
  }

  topo_order_.reserve(spls_.size());
  for (const auto v : spls_)
    topo_order_.push_back(static_cast<std::uint32_t>(v));

  std::vector<std::vector<int>> global_dependencies(opensn::mpi_comm.size());
  CommunicateLocationDependencies(location_dependencies_, global_dependencies);
  BuildTaskGraph();

  max_num_local_psi_slots_ = num_loc_cells;
  task_slot_ids_.resize(num_loc_cells);
  std::iota(task_slot_ids_.begin(), task_slot_ids_.end(), 0);
}

const std::vector<Task>&
CBC_SPDS::GetTaskList() const noexcept
{
  return task_list_;
}

void
CBC_SPDS::ComputeMaxNumLocalPsiSlots()
{
  CALI_CXX_MARK_SCOPE("CBC_SPDS::ComputeMaxNumLocalPsiSlots");

  const std::uint32_t num_tasks = static_cast<std::uint32_t>(task_list_.size());
  if (num_tasks == 0)
  {
    max_num_local_psi_slots_ = 0;
    return;
  }

  thread_local ThreadLocalWorkspace workspace;
  
  DenseHopcroftKarp allocator(num_tasks, task_list_, topo_order_, task_slot_ids_, workspace);
  max_num_local_psi_slots_ = allocator.Solve();
}

#ifndef __OPENSN_WITH_GPU__
void
CBC_SPDS::CopyTaskGraphDataOnDevice() const
{
}

void
CBC_SPDS::FreeDeviceData() const
{
}
#endif

CBC_SPDS::~CBC_SPDS()
{
  FreeDeviceData();
}

} // namespace opensn
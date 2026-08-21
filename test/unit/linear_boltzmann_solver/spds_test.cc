#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc_slot_planner.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <random>

namespace opensn
{

namespace
{

std::vector<std::vector<unsigned char>>
BuildExplicitClosure(const std::size_t num_tasks,
                     const std::vector<std::uint32_t>& successor_offsets,
                     const std::vector<std::uint32_t>& successors)
{
  std::vector<std::vector<unsigned char>> reachable(num_tasks,
                                                    std::vector<unsigned char>(num_tasks, 0));
  for (std::size_t task = 0; task < num_tasks; ++task)
  {
    reachable[task][task] = 1;
    for (auto i = successor_offsets[task]; i < successor_offsets[task + 1]; ++i)
      reachable[task][successors[i]] = 1;
  }
  for (std::size_t intermediate = 0; intermediate < num_tasks; ++intermediate)
    for (std::size_t source = 0; source < num_tasks; ++source)
      if (reachable[source][intermediate])
        for (std::size_t destination = 0; destination < num_tasks; ++destination)
          reachable[source][destination] |= reachable[intermediate][destination];
  return reachable;
}

std::size_t
ComputeExplicitSlotCount(const std::vector<std::vector<unsigned char>>& reachable,
                         const std::vector<std::uint32_t>& face_producers,
                         const std::vector<std::uint32_t>& face_consumers)
{
  constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> mate_v(face_producers.size(), invalid);
  std::vector<unsigned char> visited(face_producers.size(), 0);
  const std::function<bool(std::uint32_t)> augment = [&](const std::uint32_t u_face)
  {
    for (std::uint32_t v_face = 0; v_face < face_producers.size(); ++v_face)
    {
      if (visited[v_face] or not reachable[face_consumers[u_face]][face_producers[v_face]])
        continue;
      visited[v_face] = 1;
      if (mate_v[v_face] == invalid or augment(mate_v[v_face]))
      {
        mate_v[v_face] = u_face;
        return true;
      }
    }
    return false;
  };

  std::size_t matching_size = 0;
  for (std::uint32_t face = 0; face < face_producers.size(); ++face)
  {
    std::ranges::fill(visited, 0);
    matching_size += augment(face);
  }
  return face_producers.size() - matching_size;
}

} // namespace

TEST(CBCSPDSTest, WeightedTwoVertexFeedbackArcSet)
{
  Graph graph(2);
  boost::add_edge(0, 1, 100.0, graph);
  boost::add_edge(1, 0, 1.0, graph);

  const auto feedback_arc_set = CBC_SPDS::RemoveCyclicDependencies(graph);

  ASSERT_EQ(feedback_arc_set.size(), 1);
  const std::pair<Vertex, Vertex> expected_edge{1, 0};
  EXPECT_EQ(feedback_arc_set.front(), expected_edge);
}

TEST(CBCSPDSTest, FeedbackArcSetReversesSinkRemovalOrder)
{
  Graph graph(4);
  boost::add_edge(0, 1, 1.0, graph);
  boost::add_edge(1, 2, 1.0, graph);
  boost::add_edge(2, 3, 1.0, graph);
  boost::add_edge(3, 0, 1.0, graph);

  const auto feedback_arc_set = CBC_SPDS::RemoveCyclicDependencies(graph);

  ASSERT_EQ(feedback_arc_set.size(), 1);
  const std::pair<Vertex, Vertex> expected_edge{3, 0};
  EXPECT_EQ(feedback_arc_set.front(), expected_edge);
}

TEST(CBCSPDSTest, FeedbackArcSetDeduplicatesParallelEdges)
{
  Graph graph(2);
  boost::add_edge(0, 1, 60.0, graph);
  boost::add_edge(0, 1, 40.0, graph);
  boost::add_edge(1, 0, 0.5, graph);
  boost::add_edge(1, 0, 0.5, graph);

  const auto feedback_arc_set = CBC_SPDS::RemoveCyclicDependencies(graph);

  ASSERT_EQ(feedback_arc_set.size(), 1);
  const std::pair<Vertex, Vertex> expected_edge{1, 0};
  EXPECT_EQ(feedback_arc_set.front(), expected_edge);
}

TEST(CBCSPDSTest, ExactFeedbackArcSetImprovesHeuristicOrdering)
{
  Graph graph(5);
  boost::add_edge(0, 2, 4.0, graph);
  boost::add_edge(0, 3, 6.0, graph);
  boost::add_edge(0, 4, 1.0, graph);
  boost::add_edge(1, 3, 3.0, graph);
  boost::add_edge(3, 0, 4.0, graph);
  boost::add_edge(3, 2, 5.0, graph);
  boost::add_edge(3, 4, 3.0, graph);
  boost::add_edge(4, 1, 9.0, graph);

  const auto feedback_arc_set = CBC_SPDS::RemoveCyclicDependencies(graph);

  const std::vector<std::pair<Vertex, Vertex>> expected_edges{{3, 0}, {3, 4}};
  EXPECT_EQ(feedback_arc_set, expected_edges);
}

TEST(CBCSPDSTest, LocalFaceSlotsReuseAfterSameCellConsumption)
{
  const std::vector<std::uint32_t> successor_offsets{0, 1, 2, 2};
  const std::vector<std::uint32_t> successors{1, 2};
  const std::vector<std::uint32_t> face_producers{0, 1};
  const std::vector<std::uint32_t> face_consumers{1, 2};
  const std::vector<std::uint32_t> producer_face_offsets{0, 1, 2, 2};
  std::vector<std::uint32_t> face_slots;

  const auto num_slots = detail::ComputeLocalFaceSlotPlan(successor_offsets,
                                                          successors,
                                                          face_producers,
                                                          face_consumers,
                                                          producer_face_offsets,
                                                          face_slots);

  EXPECT_EQ(num_slots, 1);
  EXPECT_EQ(face_slots, (std::vector<std::uint32_t>{0, 0}));
}

TEST(CBCSPDSTest, LocalFaceSlotsSeparateIncomparableFaces)
{
  const std::vector<std::uint32_t> successor_offsets{0, 2, 2, 2};
  const std::vector<std::uint32_t> successors{1, 2};
  const std::vector<std::uint32_t> face_producers{0, 0};
  const std::vector<std::uint32_t> face_consumers{1, 2};
  const std::vector<std::uint32_t> producer_face_offsets{0, 2, 2, 2};
  std::vector<std::uint32_t> face_slots;

  const auto num_slots = detail::ComputeLocalFaceSlotPlan(successor_offsets,
                                                          successors,
                                                          face_producers,
                                                          face_consumers,
                                                          producer_face_offsets,
                                                          face_slots);

  EXPECT_EQ(num_slots, 2);
  ASSERT_EQ(face_slots.size(), 2);
  EXPECT_NE(face_slots[0], face_slots[1]);
}

TEST(CBCSPDSTest, LocalFaceSlotsUseTransitiveReachability)
{
  const std::vector<std::uint32_t> successor_offsets{0, 1, 2, 3, 3};
  const std::vector<std::uint32_t> successors{1, 2, 3};
  const std::vector<std::uint32_t> face_producers{0, 2};
  const std::vector<std::uint32_t> face_consumers{1, 3};
  const std::vector<std::uint32_t> producer_face_offsets{0, 1, 1, 2, 2};
  std::vector<std::uint32_t> face_slots;

  const auto num_slots = detail::ComputeLocalFaceSlotPlan(successor_offsets,
                                                          successors,
                                                          face_producers,
                                                          face_consumers,
                                                          producer_face_offsets,
                                                          face_slots);

  EXPECT_EQ(num_slots, 1);
  EXPECT_EQ(face_slots, (std::vector<std::uint32_t>{0, 0}));
}

TEST(CBCSPDSTest, LocalFaceSlotsFindExactTransitiveMatching)
{
  const std::vector<std::uint32_t> successor_offsets{0, 1, 2, 3, 5, 5, 5};
  const std::vector<std::uint32_t> successors{2, 2, 3, 4, 5};
  const std::vector<std::uint32_t> face_producers{0, 1, 2, 3, 3};
  const std::vector<std::uint32_t> face_consumers{2, 2, 3, 4, 5};
  const std::vector<std::uint32_t> producer_face_offsets{0, 1, 2, 3, 5, 5, 5};
  std::vector<std::uint32_t> face_slots;

  const auto num_slots = detail::ComputeLocalFaceSlotPlan(successor_offsets,
                                                          successors,
                                                          face_producers,
                                                          face_consumers,
                                                          producer_face_offsets,
                                                          face_slots);

  // Same-cell handoffs alone produce three chains. Transitive reachability admits a
  // maximum matching of size three and therefore the exact two-chain cover.
  EXPECT_EQ(num_slots, 2);
  ASSERT_EQ(face_slots.size(), face_producers.size());
  EXPECT_EQ(*std::ranges::max_element(face_slots), 1);
}

TEST(CBCSPDSTest, LocalFaceSlotsMatchExplicitClosure)
{
  std::mt19937 generator(0xCBCCBCCD);
  for (std::uint32_t num_tasks = 2; num_tasks <= 9; ++num_tasks)
    for (std::size_t sample = 0; sample < 250; ++sample)
    {
      std::vector<std::uint32_t> successor_offsets(num_tasks + 1);
      std::vector<std::uint32_t> successors;
      std::vector<std::uint32_t> face_producers;
      std::vector<std::uint32_t> face_consumers;
      std::vector<std::uint32_t> producer_face_offsets(num_tasks + 1);
      for (std::uint32_t producer = 0; producer < num_tasks; ++producer)
      {
        successor_offsets[producer] = static_cast<std::uint32_t>(successors.size());
        producer_face_offsets[producer] = static_cast<std::uint32_t>(face_producers.size());
        for (std::uint32_t consumer = producer + 1; consumer < num_tasks; ++consumer)
        {
          const auto multiplicity = generator() % 3;
          for (std::uint32_t face = 0; face < multiplicity; ++face)
          {
            successors.push_back(consumer);
            face_producers.push_back(producer);
            face_consumers.push_back(consumer);
          }
        }
      }
      successor_offsets.back() = static_cast<std::uint32_t>(successors.size());
      producer_face_offsets.back() = static_cast<std::uint32_t>(face_producers.size());

      const auto reachable = BuildExplicitClosure(num_tasks, successor_offsets, successors);
      const auto expected_slots =
        ComputeExplicitSlotCount(reachable, face_producers, face_consumers);
      std::vector<std::uint32_t> face_slots;
      const auto actual_slots = detail::ComputeLocalFaceSlotPlan(successor_offsets,
                                                                 successors,
                                                                 face_producers,
                                                                 face_consumers,
                                                                 producer_face_offsets,
                                                                 face_slots);

      ASSERT_EQ(actual_slots, expected_slots);
      ASSERT_EQ(face_slots.size(), face_producers.size());
      std::vector<std::uint32_t> previous_face(actual_slots,
                                               std::numeric_limits<std::uint32_t>::max());
      for (std::uint32_t face = 0; face < face_slots.size(); ++face)
      {
        const auto slot = face_slots[face];
        ASSERT_LT(slot, actual_slots);
        if (previous_face[slot] != std::numeric_limits<std::uint32_t>::max())
          EXPECT_TRUE(reachable[face_consumers[previous_face[slot]]][face_producers[face]]);
        previous_face[slot] = face;
      }
    }
}

} // namespace opensn

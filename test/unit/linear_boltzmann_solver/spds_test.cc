#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc_slot_planner.h"
#include "gtest/gtest.h"

namespace opensn
{

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

} // namespace opensn

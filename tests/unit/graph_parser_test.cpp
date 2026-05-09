#include "cycle_enum/core/graph.hpp"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace {

std::filesystem::path test_data_path(const std::string& filename) {
  return std::filesystem::path(CYCLE_ENUM_TEST_DATA_DIR) / filename;
}

TEST(GraphParserTest, GroupsEdgesSortsTimestampsAndCompactsVertices) {
  const cycle_enum::TemporalGraph graph =
      cycle_enum::read_temporal_graph(test_data_path("sample_temporal.txt"));

  ASSERT_EQ(graph.vertex_count(), 3U);
  EXPECT_EQ(graph.edge_count(), 3U);
  EXPECT_EQ(graph.timestamp_count(), 4U);

  ASSERT_TRUE(graph.compact_id(10).has_value());
  ASSERT_TRUE(graph.compact_id(20).has_value());
  ASSERT_TRUE(graph.compact_id(30).has_value());
  EXPECT_EQ(graph.external_id(*graph.compact_id(10)), 10);

  const auto source_10 = *graph.compact_id(10);
  const auto source_20 = *graph.compact_id(20);
  const auto source_30 = *graph.compact_id(30);

  const cycle_enum::TemporalEdge* edge_10_20 =
      graph.find_edge(source_10, source_20);
  ASSERT_NE(edge_10_20, nullptr);
  EXPECT_EQ(edge_10_20->timestamps, (std::vector<cycle_enum::Timestamp>{3, 5}));

  const cycle_enum::TemporalEdge* edge_20_10 =
      graph.find_edge(source_20, source_10);
  ASSERT_NE(edge_20_10, nullptr);
  EXPECT_EQ(edge_20_10->timestamps, (std::vector<cycle_enum::Timestamp>{7}));

  const cycle_enum::TemporalEdge* edge_30_10 =
      graph.find_edge(source_30, source_10);
  ASSERT_NE(edge_30_10, nullptr);
  EXPECT_EQ(edge_30_10->timestamps, (std::vector<cycle_enum::Timestamp>{6}));
}

TEST(GraphParserTest, SkipsSelfLoopsWithoutInterningLoopOnlyVertex) {
  const cycle_enum::TemporalGraph graph =
      cycle_enum::read_temporal_graph(test_data_path("sample_temporal.txt"));

  EXPECT_FALSE(graph.compact_id(40).has_value());
}

TEST(GraphParserTest, AssignsCompactIdsInAscendingExternalOrder) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "cycle_enum_unsorted_ids.txt";
  {
    std::ofstream output(path);
    output << "50 10 1\n";
    output << "10 30 2\n";
  }

  const cycle_enum::TemporalGraph graph =
      cycle_enum::read_temporal_graph(path);

  ASSERT_TRUE(graph.compact_id(10).has_value());
  ASSERT_TRUE(graph.compact_id(30).has_value());
  ASSERT_TRUE(graph.compact_id(50).has_value());
  EXPECT_EQ(*graph.compact_id(10), 0U);
  EXPECT_EQ(*graph.compact_id(30), 1U);
  EXPECT_EQ(*graph.compact_id(50), 2U);
  EXPECT_EQ(graph.external_id(0), 10);
  EXPECT_EQ(graph.external_id(1), 30);
  EXPECT_EQ(graph.external_id(2), 50);

  std::filesystem::remove(path);
}

TEST(GraphParserTest, ThrowsForMissingFile) {
  EXPECT_THROW(
      {
        const cycle_enum::TemporalGraph graph =
            cycle_enum::read_temporal_graph(test_data_path("missing.txt"));
        (void)graph;
      },
      cycle_enum::GraphParseError);
}

TEST(GraphParserTest, ThrowsForMalformedRows) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "cycle_enum_bad_graph.txt";
  {
    std::ofstream output(path);
    output << "1 2\n";
  }

  EXPECT_THROW(
      {
        const cycle_enum::TemporalGraph graph =
            cycle_enum::read_temporal_graph(path);
        (void)graph;
      },
      cycle_enum::GraphParseError);

  std::filesystem::remove(path);
}

}  // namespace

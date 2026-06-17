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
  const std::filesystem::path single_field_path =
      std::filesystem::temp_directory_path() / "cycle_enum_single_field.txt";
  {
    std::ofstream output(single_field_path);
    output << "5\n";
  }
  EXPECT_THROW(
      {
        const cycle_enum::TemporalGraph graph =
            cycle_enum::read_temporal_graph(single_field_path);
        (void)graph;
      },
      cycle_enum::GraphParseError);
  std::filesystem::remove(single_field_path);

  const std::filesystem::path extra_field_path =
      std::filesystem::temp_directory_path() / "cycle_enum_extra_field.txt";
  {
    std::ofstream output(extra_field_path);
    output << "1 2 3 4\n";
  }
  EXPECT_THROW(
      {
        const cycle_enum::TemporalGraph graph =
            cycle_enum::read_temporal_graph(extra_field_path);
        (void)graph;
      },
      cycle_enum::GraphParseError);
  std::filesystem::remove(extra_field_path);

  const std::filesystem::path bad_timestamp_path =
      std::filesystem::temp_directory_path() / "cycle_enum_bad_timestamp.txt";
  {
    std::ofstream output(bad_timestamp_path);
    output << "1 2 notanumber\n";
  }
  EXPECT_THROW(
      {
        const cycle_enum::TemporalGraph graph =
            cycle_enum::read_temporal_graph(bad_timestamp_path);
        (void)graph;
      },
      cycle_enum::GraphParseError);
  std::filesystem::remove(bad_timestamp_path);
}

TEST(GraphParserTest, DefaultsTimestampWhenColumnOmitted) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "cycle_enum_no_timestamp.txt";
  {
    std::ofstream output(path);
    output << "1 2\n";
    output << "2 1\n";
  }

  const cycle_enum::TemporalGraph graph =
      cycle_enum::read_temporal_graph(path);

  ASSERT_EQ(graph.vertex_count(), 2U);
  ASSERT_TRUE(graph.compact_id(1).has_value());
  ASSERT_TRUE(graph.compact_id(2).has_value());

  const cycle_enum::TemporalEdge* edge =
      graph.find_edge(*graph.compact_id(1), *graph.compact_id(2));
  ASSERT_NE(edge, nullptr);
  EXPECT_EQ(edge->timestamps, (std::vector<cycle_enum::Timestamp>{0}));

  std::filesystem::remove(path);
}

TEST(GraphParserTest, ParsesCommaSeparatedRowsWithAndWithoutTimestamp) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "cycle_enum_comma.txt";
  {
    std::ofstream output(path);
    output << "% directed unweighted\n";
    output << "1,2,7\n";
    output << "2,1\n";
  }

  const cycle_enum::TemporalGraph graph =
      cycle_enum::read_temporal_graph(path);

  ASSERT_EQ(graph.vertex_count(), 2U);

  const cycle_enum::TemporalEdge* edge_1_2 =
      graph.find_edge(*graph.compact_id(1), *graph.compact_id(2));
  ASSERT_NE(edge_1_2, nullptr);
  EXPECT_EQ(edge_1_2->timestamps, (std::vector<cycle_enum::Timestamp>{7}));

  const cycle_enum::TemporalEdge* edge_2_1 =
      graph.find_edge(*graph.compact_id(2), *graph.compact_id(1));
  ASSERT_NE(edge_2_1, nullptr);
  EXPECT_EQ(edge_2_1->timestamps, (std::vector<cycle_enum::Timestamp>{0}));

  std::filesystem::remove(path);
}

TEST(GraphParserTest, ReadsMatrixMarketAsStoredDirectedEdges) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "cycle_enum_mm.mtx";
  {
    std::ofstream output(path);
    output << "%%MatrixMarket matrix coordinate pattern symmetric\n";
    output << "% a comment line\n";
    output << "3 3 2\n";  // dimensions header: rows cols nnz (must be skipped)
    output << "1 2\n";
    output << "2 3\n";
  }

  const cycle_enum::TemporalGraph graph =
      cycle_enum::read_temporal_graph(path);

  // Exactly two directed edges, no symmetrization: 1->2 and 2->3 only.
  ASSERT_EQ(graph.vertex_count(), 3U);
  EXPECT_EQ(graph.edge_count(), 2U);
  EXPECT_NE(graph.find_edge(*graph.compact_id(1), *graph.compact_id(2)),
            nullptr);
  EXPECT_NE(graph.find_edge(*graph.compact_id(2), *graph.compact_id(3)),
            nullptr);
  EXPECT_EQ(graph.find_edge(*graph.compact_id(2), *graph.compact_id(1)),
            nullptr);

  std::filesystem::remove(path);
}

}  // namespace

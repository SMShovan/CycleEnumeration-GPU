#include "cycle_enum/core/graph.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

namespace cycle_enum {

namespace {

bool is_ignored_line(const std::string& line) {
  const std::size_t first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return true;
  }
  return line[first] == '#' || line[first] == '%';
}

std::string line_error(const std::filesystem::path& path,
                       const std::size_t line_number,
                       const std::string& reason) {
  std::ostringstream message;
  message << path.string() << ':' << line_number << ": " << reason;
  return message.str();
}

struct RawTemporalEdge {
  ExternalVertexId source = 0;
  ExternalVertexId target = 0;
  Timestamp timestamp = 0;
};

}  // namespace

GraphParseError::GraphParseError(const std::string& message)
    : std::runtime_error(message) {}

TemporalGraph::TemporalGraph(std::vector<ExternalVertexId> external_ids,
                             std::vector<TemporalEdge> edges)
    : external_ids_(std::move(external_ids)), edges_(std::move(edges)) {
  for (const TemporalEdge& edge : edges_) {
    timestamp_count_ += edge.timestamps.size();
  }
}

std::size_t TemporalGraph::vertex_count() const noexcept {
  return external_ids_.size();
}

std::size_t TemporalGraph::edge_count() const noexcept {
  return edges_.size();
}

std::size_t TemporalGraph::timestamp_count() const noexcept {
  return timestamp_count_;
}

const std::vector<TemporalEdge>& TemporalGraph::edges() const noexcept {
  return edges_;
}

ExternalVertexId TemporalGraph::external_id(const VertexId vertex) const {
  if (vertex >= external_ids_.size()) {
    throw std::out_of_range("compact vertex id is out of range");
  }
  return external_ids_[vertex];
}

std::optional<VertexId> TemporalGraph::compact_id(
    const ExternalVertexId external) const noexcept {
  const auto found =
      std::find(external_ids_.begin(), external_ids_.end(), external);
  if (found == external_ids_.end()) {
    return std::nullopt;
  }
  return static_cast<VertexId>(std::distance(external_ids_.begin(), found));
}

const TemporalEdge* TemporalGraph::find_edge(const VertexId source,
                                             const VertexId target) const
    noexcept {
  const auto found = std::find_if(
      edges_.begin(), edges_.end(), [source, target](const TemporalEdge& edge) {
        return edge.source == source && edge.target == target;
      });
  return found == edges_.end() ? nullptr : &*found;
}

TemporalGraph read_temporal_graph(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw GraphParseError("failed to open temporal graph file: " +
                          path.string());
  }

  std::vector<RawTemporalEdge> raw_edges;
  std::set<ExternalVertexId> external_id_set;

  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (is_ignored_line(line)) {
      continue;
    }

    std::istringstream row(line);
    ExternalVertexId source_external = 0;
    ExternalVertexId target_external = 0;
    Timestamp timestamp = 0;
    std::string trailing;

    if (!(row >> source_external >> target_external >> timestamp) ||
        (row >> trailing)) {
      throw GraphParseError(
          line_error(path, line_number,
                     "expected exactly three fields: source target timestamp"));
    }

    if (source_external == target_external) {
      continue;
    }

    raw_edges.push_back(
        RawTemporalEdge{source_external, target_external, timestamp});
    external_id_set.insert(source_external);
    external_id_set.insert(target_external);
  }

  std::vector<ExternalVertexId> external_ids(external_id_set.begin(),
                                             external_id_set.end());
  std::unordered_map<ExternalVertexId, VertexId> compact_ids;
  compact_ids.reserve(external_ids.size());
  for (std::size_t index = 0; index < external_ids.size(); ++index) {
    compact_ids.emplace(external_ids[index], static_cast<VertexId>(index));
  }

  std::map<std::pair<VertexId, VertexId>, std::vector<Timestamp>> grouped_edges;
  for (const RawTemporalEdge& raw_edge : raw_edges) {
    const VertexId source = compact_ids.at(raw_edge.source);
    const VertexId target = compact_ids.at(raw_edge.target);
    grouped_edges[{source, target}].push_back(raw_edge.timestamp);
  }

  std::vector<TemporalEdge> edges;
  edges.reserve(grouped_edges.size());
  for (auto& [endpoints, timestamps] : grouped_edges) {
    std::sort(timestamps.begin(), timestamps.end());
    edges.push_back(TemporalEdge{
        endpoints.first,
        endpoints.second,
        std::move(timestamps),
    });
  }

  return TemporalGraph(std::move(external_ids), std::move(edges));
}

}  // namespace cycle_enum

#include "cycle_enum/core/graph.hpp"

#include <algorithm>
#include <fstream>
#include <map>
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

VertexId intern_vertex(
    const ExternalVertexId external,
    std::unordered_map<ExternalVertexId, VertexId>& compact_ids,
    std::vector<ExternalVertexId>& external_ids) {
  const auto existing = compact_ids.find(external);
  if (existing != compact_ids.end()) {
    return existing->second;
  }

  const VertexId compact = static_cast<VertexId>(external_ids.size());
  compact_ids.emplace(external, compact);
  external_ids.push_back(external);
  return compact;
}

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

  std::unordered_map<ExternalVertexId, VertexId> compact_ids;
  std::vector<ExternalVertexId> external_ids;
  std::map<std::pair<VertexId, VertexId>, std::vector<Timestamp>> grouped_edges;

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

    const VertexId source =
        intern_vertex(source_external, compact_ids, external_ids);
    const VertexId target =
        intern_vertex(target_external, compact_ids, external_ids);
    grouped_edges[{source, target}].push_back(timestamp);
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


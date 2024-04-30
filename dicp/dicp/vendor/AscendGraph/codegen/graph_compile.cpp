#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ge_builder.h"
#include "ge_graph.h"
#include "ge_runner.h"
#include "graph_utils.h"

extern "C" {

std::unordered_map<int, std::shared_ptr<GEGraph>> graph_manager;
std::unique_ptr<GEGraphRunner> graph_runner;

void init(int device_id, const char* config_file_path) {
  graph_runner = std::make_unique<GEGraphRunner>(device_id, config_file_path);
  std::cout << "graph runner init success!" << std::endl;
}

void release() { graph_runner.reset(); }

void add_graph(int graph_id, const char* graph_json_file,
               const char* graph_key) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());

  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);

  std::vector<Tensor> input_tensors;
  buildGraph(graph, graph_json, input_tensors);

  auto graph_spec = graph_runner->addGraph(graph_id, graph, graph_key);
  auto acl_graph = std::make_shared<GEGraph>(graph_id, graph_key, graph,
                                             graph_spec, input_tensors);
  graph_manager[graph_id] = std::move(acl_graph);
}

size_t get_const_size(int graph_id) {
  return graph_manager[graph_id]->const_memory_size();
}

size_t get_workspace_size(int graph_id) {
  return graph_manager[graph_id]->feature_memory_size();
}

std::string get_shapes(const std::vector<std::vector<int64_t>>& shapes) {
  std::ostringstream oss;
  for (size_t i = 0; i < shapes.size(); ++i) {
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      oss << shapes[i][j] << (j != shapes[i].size() - 1 ? "," : "");
    }
    oss << (i != shapes.size() - 1 ? ";" : "");
  }
  return oss.str();
}

void get_input_shapes(int graph_id, char* input_shapes) {
  std::string str = get_shapes(graph_manager[graph_id]->get_input_shapes());
  strncpy(input_shapes, str.c_str(), str.size() + 1);
}

void get_output_shapes(int graph_id, char* output_shapes) {
  std::string str = get_shapes(graph_manager[graph_id]->get_output_shapes());
  strncpy(output_shapes, str.c_str(), str.size() + 1);
}

std::string get_dtypes(const std::vector<int>& dtypes) {
  std::ostringstream oss;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    oss << dtypes[i] << (i != dtypes.size() - 1 ? ";" : "");
  }
  return oss.str();
}

void get_input_dtypes(int graph_id, char* input_dtypes) {
  std::string str = get_dtypes(graph_manager[graph_id]->get_input_dtypes());
  strncpy(input_dtypes, str.c_str(), str.size() + 1);
}

void get_output_dtypes(int graph_id, char* output_dtypes) {
  std::string str = get_dtypes(graph_manager[graph_id]->get_output_dtypes());
  strncpy(output_dtypes, str.c_str(), str.size() + 1);
}

void set_graph_memory(int graph_id, void* const_mem_ptr, void* workspace_ptr,
                      size_t const_size, size_t workspace_size) {
  graph_runner->setConstMem(graph_id, const_mem_ptr, const_size);
  graph_runner->setWorkSpace(graph_id, workspace_ptr, workspace_size);
}

void run(int graph_id, void* stream, void* inputs_data[], void* outputs_data[],
         int64_t inputs_data_size[], int64_t outputs_data_size[]) {
  graph_manager[graph_id]->set_input_output_data(
      inputs_data, outputs_data, inputs_data_size, outputs_data_size);
  graph_runner->runGraphWithStreamAsync(graph_manager[graph_id], stream);
}

void compile_and_save(const char* graph_path, const char* graph_json_file,
                      const char* fusion_switch_file,
                      const char* global_options_file) {
  std::string graph_name = "BuildGraph";
  Graph graph(graph_name.c_str());
  std::ifstream f(graph_json_file);
  json graph_json = json::parse(f);

  std::vector<Tensor> input_tensors;
  buildGraph(graph, graph_json, input_tensors);

  std::map<AscendString, AscendString> options;
  GEGraphBuilder builder{fusion_switch_file, global_options_file};
  builder.saveGraph(graph_path, graph, options);
}
}

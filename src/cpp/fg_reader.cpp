#include "fg_reader.h"

using namespace std;

namespace fg{
void parse_timestamps(const torch::jit::script::Module fg_container,
        std::vector<double> &timestamps){
    torch::Tensor tStamps = fg_container.attr("tstamp").toTensor();
    auto dim = tStamps.dim();

    for (int i = 0; i < tStamps.size(0); i++){
        timestamps[0] = tStamps[i].item<double>();
    }
}

void parse_weights(const int index, std::vector<std::vector<double>> m){

}

void parse_factor_graph_file(const std::string filename, std::vector<double> &timestamps){
  torch::jit::script::Module container = torch::jit::load(filename);
  torch::Tensor weights = container.attr("c_map").toTensor();
  parse_timestamps(container, timestamps);
}
};

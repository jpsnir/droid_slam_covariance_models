#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>


namespace fg{
void parse_timestamps(const torch::jit::script::Module fg_container,
        std::vector<double> &timestamps);
void parse_weights(const int index, std::vector<std::vector<double>> m);
void parse_factor_graph_file(const std::string filename, std::vector<double> &timestamps);

};

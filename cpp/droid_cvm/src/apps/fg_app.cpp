#include <iostream>
#include <vector>
#include <string>
#include "fg_parser/fg_reader.h"
#include "fg_parser/fg_builder.h"
#include<custom_factors/gen/cpp/symforce/sym/droid_slam_residual_single_factor.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ./fg_cov <factograph.pt> <factor_graph.pt> ... "
                 "<factor_graph.pt> \n";
    exit(1);
  }

  std::vector<std::string> filepaths(5);
  for (int i = 1; i <= argc - 1; i++) {
    filepaths[i - 1] = *(argv + i);
    std::cout << *(argv + i) << std::endl;
  }
  std::vector<double> timestamps(100);
  fg::parse_factor_graph_file(filepaths[0], timestamps);
  fg::construct_factor_graph(timestamps);
}

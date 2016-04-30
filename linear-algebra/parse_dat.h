#include <fstream>
#include <string>
#include <vector>

// Parse airfoil data
std::pair<std::vector<double>, std::vector<double>>
parse_dat(std::string path) {
  std::ifstream buffer(path);
  std::vector<double> x;
  std::vector<double> y;

  while (!buffer.eof()) {
    double tmp;
    buffer >> tmp;

    if (buffer.peek() == '\n') {
      y.push_back(tmp);
    } else {
      x.push_back(tmp);
    }
  }

  return {x, y};
}

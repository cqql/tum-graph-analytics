#ifndef GAML_MF_ALS_LOGGER_H_
#define GAML_MF_ALS_LOGGER_H_

#include <iostream>

namespace gaml {

namespace mf {

namespace als {

class Logger {
 public:
  void log(int iteration, float mse, float aimprov, float rimprov);
};
}
}
}

#endif  // GAML_MF_ALS_LOGGER_H_

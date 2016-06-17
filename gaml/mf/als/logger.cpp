#include "logger.h"

namespace gaml {

namespace mf {

namespace als {

void Logger::log(int iteration, float mse, float aimprov, float rimprov) {
  std::cout << "Iteration " << iteration << ": MSE = " << mse
            << ", AImp = " << aimprov << ", RImp = " << rimprov << std::endl;
}
}
}
}

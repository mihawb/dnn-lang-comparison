#ifndef _BUILDERS_H_
#define _BUILDERS_H_

#include "network.h"

namespace cudl {

void FullyConnectedNetBuilder(Network &model);

void SimpleConvNetBuilder(Network &model);

void ModelFactory(Network &model, int m);

}

#endif // _BUILDERS_H_
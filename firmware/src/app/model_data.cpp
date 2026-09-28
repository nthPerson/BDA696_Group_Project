// Pulls the generated int8 model (firmware/model/gate_model_data.cc) into the build.
// Regenerate with `formcoach train gate --model cnn`; never edit the .cc by hand.
#ifndef FC_GATE_ENERGY
#include "../../model/gate_model_data.cc"
#endif

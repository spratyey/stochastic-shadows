#include <gflags/gflags.h>

DEFINE_bool(isInteractive, true, "Whether to render interactively");
DEFINE_string(scenePath, "../scenes/scene_configs/bistro.json", "Path to the scene file");
DEFINE_string(outputPath, "../results/", "Path to output directory");
DEFINE_uint32(samples, 1000, "Number of samples");
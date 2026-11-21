#include "visualizer.hpp"

#include <cstdlib>

#ifndef RECAST_SAMPLE_SCENE
#define RECAST_SAMPLE_SCENE ""
#endif

int main(int argc, char** argv) {
	foto::ApplicationConfig config = {
		.title = "foto",
		.width = 1280,
		.height = 720
	};

	try {
		std::string scenePath = argc > 1 ? argv[1] : RECAST_SAMPLE_SCENE;

		Visualizer app(config, scenePath);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

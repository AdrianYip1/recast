#include "visualizer.hpp"

#include <cstdlib>

#ifndef RECAST_SAMPLE_SCENE
#define RECAST_SAMPLE_SCENE ""
#endif

int main(int argc, char** argv) {
	foto::ApplicationConfig config = {
		.title = "ReCast",
		.width = 1280,
		.height = 720
	};

	try {
		// no scene given: start on the welcome screen, which asks for a scene photo
		std::string scenePath = argc > 1 ? argv[1] : "";

		Visualizer app(config, scenePath);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

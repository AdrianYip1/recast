#include "visualizer.hpp"

#include <cstdlib>

int main(int argc, char** argv) {
	foto::ApplicationConfig config = {
		.title = "foto",
		.width = 1280,
		.height = 720
	};

	try {
		Visualizer app(config, argc > 1 ? argv[1] : "");
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

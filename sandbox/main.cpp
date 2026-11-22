#include "visualizer.hpp"

#include <cstdlib>
#include <filesystem>

#include "pipeline/pythonJob.hpp"

#ifndef RECAST_SAMPLE_SCENE
#define RECAST_SAMPLE_SCENE ""
#endif

namespace {
	void setEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
		_putenv_s(name, value.c_str());
#else
		setenv(name, value.c_str(), 1);
#endif
	}
} // namespace

int main(int argc, char** argv) {
	// the release zip: shaders load from next to the exe wherever it's started from, and Python uses the bundled
	// models without going online
	if (recast::portableInstall()) {
		std::error_code error;
		std::filesystem::current_path(recast::executableDir(), error);
		setEnvironment("HF_HUB_CACHE", (recast::executableDir() / "models").string());
		setEnvironment("HF_HUB_OFFLINE", "1");
		setEnvironment("PYTHONNOUSERSITE", "1");
	}

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

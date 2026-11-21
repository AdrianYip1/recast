#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace recast {
	// One Python step at a time, on a background thread: either a shell command, or a request handed to the
	// long-running worker (python -m recast.worker), which keeps torch and the models loaded between steps
	class PythonJob {
	public:
		PythonJob() = default;
		PythonJob(const PythonJob&) = delete;
		PythonJob& operator=(const PythonJob&) = delete;
		~PythonJob();

		void start(const std::string& command);
		// Writes requestJson into the worker's queue and waits for its .done file; fails if workerAlive turns false
		void startInWorker(const std::filesystem::path& queue, const std::string& requestJson, std::function<bool()> workerAlive);
		bool running() const { return busy; }
		// true once, on the first call after the step ends
		bool finished();
		int exitCode() const { return exit; }
		std::string output() const;

	private:
		std::thread thread;
		mutable std::mutex mutex;
		std::string log;
		std::atomic<bool> busy = false;
		std::atomic<bool> done = false;
		std::atomic<bool> cancel = false;
		std::atomic<int> exit = 0;

		void begin();
		void append(const std::string& text);
	};

	// The worker process: started once, stopped when the engine closes (or it notices the engine is gone)
	class PythonWorker {
	public:
		PythonWorker() = default;
		PythonWorker(const PythonWorker&) = delete;
		PythonWorker& operator=(const PythonWorker&) = delete;
		~PythonWorker();

		void start(const std::string& command, const std::filesystem::path& queue);
		bool alive() const { return state && state->running; }
		const std::filesystem::path& queue() const { return queueDir; }

	private:
		struct State {
			std::atomic<bool> running = false;
		};
		std::shared_ptr<State> state;
		std::filesystem::path queueDir;
	};

	int currentProcessId();

	// Native open-file dialog, "" when cancelled. filter uses the Windows format: "Name\0*.jpg;*.png\0"
	// Windows only for now, elsewhere it returns "" and the path is typed in
	std::string openFileDialog(const char* title, const char* filter);
} // namespace recast

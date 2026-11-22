#include "pipeline/pythonJob.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#define popen _popen
#define pclose _pclose
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace recast {
	namespace {
		std::string readFile(const std::filesystem::path& path) {
			std::ifstream file(path, std::ios::binary);
			std::ostringstream text;
			text << file.rdbuf();
			return text.str();
		}
	}

	PythonJob::~PythonJob() {
		cancel = true;
		if (thread.joinable()) thread.join();
	}

	void PythonJob::begin() {
		if (thread.joinable()) thread.join();
		{
			std::lock_guard lock(mutex);
			log.clear();
		}
		exit = -1;
		done = false;
		busy = true;
	}

	void PythonJob::append(const std::string& text) {
		std::lock_guard lock(mutex);
		log += text;
	}

	void PythonJob::start(const std::string& command) {
		if (busy) return;
		begin();
		thread = std::thread([this, command]() {
			int status = -1;
			if (FILE* pipe = popen(command.c_str(), "r")) {
				char buffer[512];
				while (std::fgets(buffer, sizeof(buffer), pipe)) append(buffer);
				status = pclose(pipe);
#ifndef _WIN32
				if (status != -1 && WIFEXITED(status)) status = WEXITSTATUS(status);
#endif
			}
			else {
				append("couldn't start: " + command + "\n");
			}
			exit = status;
			busy = false;
			done = true;
		});
	}

	void PythonJob::startInWorker(const std::filesystem::path& queue, const std::string& requestJson, std::function<bool()> workerAlive) {
		if (busy) return;
		begin();
		static std::atomic<int> counter = 0;
		char id[32];
		std::snprintf(id, sizeof(id), "%06d", ++counter);
		thread = std::thread([this, queue, requestJson, workerAlive, name = std::string(id)]() {
			std::filesystem::path request = queue / (name + ".json"), logPath = queue / (name + ".log"), donePath = queue / (name + ".done");
			std::error_code error;
			{
				std::ofstream file(queue / (name + ".tmp"), std::ios::binary);
				file << requestJson;
			}
			std::filesystem::rename(queue / (name + ".tmp"), request, error); // atomic: the worker never sees half a request

			int status = -1;
			while (!cancel) {
				if (std::filesystem::exists(donePath, error)) {
					status = std::atoi(readFile(donePath).c_str());
					break;
				}
				if (!workerAlive()) {
					append("the Python worker stopped\n");
					break;
				}
				std::string text = readFile(logPath);
				{
					std::lock_guard lock(mutex);
					if (text.size() > log.size()) log = text;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			std::string text = readFile(logPath);
			{
				std::lock_guard lock(mutex);
				if (text.size() >= log.size()) log = text;
			}
			for (const auto& path : { request, logPath, donePath }) std::filesystem::remove(path, error);
			exit = status;
			busy = false;
			done = true;
		});
	}

	bool PythonJob::finished() {
		if (!done.exchange(false)) return false;
		if (thread.joinable()) thread.join();
		return true;
	}

	std::string PythonJob::output() const {
		std::lock_guard lock(mutex);
		return log;
	}

	PythonWorker::~PythonWorker() {
		// the worker also exits on its own once this process is gone; the stop file just makes it prompt
		if (!queueDir.empty()) std::ofstream(queueDir / "stop") << "";
	}

	void PythonWorker::start(const std::string& command, const std::filesystem::path& queue) {
		queueDir = queue;
		std::error_code error;
		std::filesystem::remove_all(queue, error);
		std::filesystem::create_directories(queue, error);
		state = std::make_shared<State>();
		state->running = true;
		// detached: it lives as long as the worker, and only touches the shared state
		std::thread([state = state, command]() {
			if (FILE* pipe = popen(command.c_str(), "r")) {
				char buffer[512];
				while (std::fgets(buffer, sizeof(buffer), pipe)) std::fputs(buffer, stdout);
				pclose(pipe);
			}
			state->running = false;
		}).detach();
	}

	int currentProcessId() {
#ifdef _WIN32
		return static_cast<int>(GetCurrentProcessId());
#else
		return static_cast<int>(getpid());
#endif
	}

	std::filesystem::path executableDir() {
#ifdef _WIN32
		wchar_t buffer[MAX_PATH * 4] = {};
		DWORD length = GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
		return std::filesystem::path(std::wstring(buffer, length)).parent_path();
#else
		std::error_code error;
		return std::filesystem::read_symlink("/proc/self/exe", error).parent_path();
#endif
	}

	bool portableInstall() {
		static const bool portable = [] {
			std::error_code error;
			return std::filesystem::exists(executableDir() / "runtime" / "python.exe", error);
		}();
		return portable;
	}

	std::string openFileDialog(const char* title, const char* filter, const char* folder) {
#ifdef _WIN32
		char path[MAX_PATH] = "";
		OPENFILENAMEA dialog = {};
		dialog.lStructSize = sizeof(dialog);
		dialog.lpstrFilter = filter;
		dialog.lpstrFile = path;
		dialog.nMaxFile = MAX_PATH;
		dialog.lpstrTitle = title;
		dialog.lpstrInitialDir = folder;
		// NOCHANGEDIR: the engine loads shaders relative to the working directory
		dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
		if (GetOpenFileNameA(&dialog)) return path;
#else
		(void)title;
		(void)filter;
		(void)folder;
#endif
		return {};
	}

	std::string saveFileDialog(const char* title, const char* filter, const char* extension, const std::string& folder, const std::string& name) {
#ifdef _WIN32
		char path[MAX_PATH] = "";
		std::snprintf(path, sizeof(path), "%s", name.c_str());
		OPENFILENAMEA dialog = {};
		dialog.lStructSize = sizeof(dialog);
		dialog.lpstrFilter = filter;
		dialog.lpstrFile = path;
		dialog.nMaxFile = MAX_PATH;
		dialog.lpstrTitle = title;
		dialog.lpstrDefExt = extension;
		dialog.lpstrInitialDir = folder.empty() ? nullptr : folder.c_str();
		dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
		if (GetSaveFileNameA(&dialog)) return path;
#else
		(void)title;
		(void)filter;
		(void)extension;
		(void)folder;
		(void)name;
#endif
		return {};
	}
} // namespace recast

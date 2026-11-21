#pragma once

#include <chrono>
#include <filesystem>
#include <system_error>

namespace foto {
	// changed() fires once per modification, after the file has stopped changing for settleTime
	// (so half-written files from non-atomic writers aren't picked up)
	class FileWatcher {
	public:
		FileWatcher() = default;
		explicit FileWatcher(std::filesystem::path filePath, std::chrono::milliseconds settleTime = std::chrono::milliseconds(200)) :
			path(std::move(filePath)), settle(settleTime), lastWrite(currentWriteTime()) {}

		bool changed() {
			if (path.empty()) return false;

			std::filesystem::file_time_type now = currentWriteTime();
			if (now != lastWrite) {
				lastWrite = now;
				pending = true;
				pendingSince = std::chrono::steady_clock::now();
				return false;
			}

			if (pending && std::chrono::steady_clock::now() - pendingSince >= settle) {
				pending = false;
				return true;
			}
			return false;
		}

		const std::filesystem::path& getPath() const { return path; };

	private:
		std::filesystem::path path;
		std::chrono::milliseconds settle{ 200 };
		std::filesystem::file_time_type lastWrite;
		bool pending = false;
		std::chrono::steady_clock::time_point pendingSince;

		std::filesystem::file_time_type currentWriteTime() const {
			std::error_code error;
			std::filesystem::file_time_type time = std::filesystem::last_write_time(path, error);
			return error ? lastWrite : time;
		}
	};
} // namespace foto

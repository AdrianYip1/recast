#include "cameraMatch.hpp"

#include <algorithm>
#include <cmath>

namespace {
	struct Plane {
		int width = 0, height = 0;
		std::vector<float> values;

		float at(int x, int y) const {
			x = std::clamp(x, 0, width - 1);
			y = std::clamp(y, 0, height - 1);
			return values[static_cast<size_t>(y) * width + x];
		}

		float bilinear(float x, float y) const {
			int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
			float fx = x - x0, fy = y - y0;
			return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
		}
	};

	float median(std::vector<float> values) {
		if (values.empty()) return 0.0f;
		std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
		return values[values.size() / 2];
	}
}

recast::CameraMatch recast::estimateCameraMatch(const LinearImage& photo) {
	CameraMatch match;
	if (photo.width < 16 || photo.height < 16) return match;

	Plane plane{ photo.width, photo.height, std::vector<float>(static_cast<size_t>(photo.width) * photo.height) };
	for (int y = 0; y < photo.height; y++) {
		for (int x = 0; x < photo.width; x++) {
			enginemath::Vec3 c = photo.at(x, y);
			plane.values[static_cast<size_t>(y) * photo.width + x] = std::sqrt(std::max(0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z, 0.0f));
		}
	}

	std::vector<float> gradient(plane.values.size(), 0.0f), response(plane.values.size(), 0.0f);
	for (int y = 1; y < photo.height - 1; y++) {
		for (int x = 1; x < photo.width - 1; x++) {
			float gx = plane.at(x + 1, y - 1) + 2 * plane.at(x + 1, y) + plane.at(x + 1, y + 1) - plane.at(x - 1, y - 1) - 2 * plane.at(x - 1, y) - plane.at(x - 1, y + 1);
			float gy = plane.at(x - 1, y + 1) + 2 * plane.at(x, y + 1) + plane.at(x + 1, y + 1) - plane.at(x - 1, y - 1) - 2 * plane.at(x, y - 1) - plane.at(x + 1, y - 1);
			size_t i = static_cast<size_t>(y) * photo.width + x;
			gradient[i] = std::sqrt(gx * gx + gy * gy);
			response[i] = std::abs(plane.at(x - 1, y - 1) - 2 * plane.at(x, y - 1) + plane.at(x + 1, y - 1)
								   - 2 * plane.at(x - 1, y) + 4 * plane.at(x, y) - 2 * plane.at(x + 1, y)
								   + plane.at(x - 1, y + 1) - 2 * plane.at(x, y + 1) + plane.at(x + 1, y + 1));
		}
	}

	// Grain: the noise mask's response on the flattest 80% of the photo
	std::vector<float> sorted = gradient;
	std::nth_element(sorted.begin(), sorted.begin() + sorted.size() * 8 / 10, sorted.end());
	float flatLimit = sorted[sorted.size() * 8 / 10];
	double sum = 0.0;
	size_t count = 0;
	for (int y = 1; y < photo.height - 1; y++) {
		for (int x = 1; x < photo.width - 1; x++) {
			size_t i = static_cast<size_t>(y) * photo.width + x;
			if (gradient[i] > flatLimit) continue;
			sum += response[i];
			count++;
		}
	}
	// Measured on brightness, which averages the channels' independent noise: 0.747 = |(0.2126, 0.7152, 0.0722)|
	if (count > 0) match.grain = static_cast<float>(std::sqrt(3.14159265 / 2.0) / 6.0 * sum / count / 0.747);

	// Blur: second moment of the brightness change across the strongest edges, minus what pixel sampling adds
	std::nth_element(sorted.begin(), sorted.begin() + sorted.size() * 995 / 1000, sorted.end());
	float edgeLimit = std::max(sorted[sorted.size() * 995 / 1000], 1e-4f);
	// Average the step profile across many edges (each scaled to a unit step) so noise cancels, then take its spread
	constexpr int HALF = 16;
	std::vector<double> profile(2 * HALF + 1, 0.0);
	int edges = 0;
	float lumGrain = match.grain * 0.747f;
	for (int y = 10; y < photo.height - 10; y += 2) {
		for (int x = 10; x < photo.width - 10; x += 2) {
			size_t i = static_cast<size_t>(y) * photo.width + x;
			if (gradient[i] < edgeLimit) continue;
			float gx = plane.at(x + 1, y) - plane.at(x - 1, y);
			float gy = plane.at(x, y + 1) - plane.at(x, y - 1);
			float length = std::sqrt(gx * gx + gy * gy);
			if (length < 1e-6f) continue;
			gx /= length;
			gy /= length;

			float contrast = plane.bilinear(x + gx * 8.0f, y + gy * 8.0f) - plane.bilinear(x - gx * 8.0f, y - gy * 8.0f);
			if (contrast < std::max(10.0f * lumGrain, 1e-3f)) continue;
			for (int k = -HALF; k <= HALF; k++) {
				float offset = k * 0.5f;
				float step = plane.bilinear(x + gx * (offset + 0.25f), y + gy * (offset + 0.25f)) - plane.bilinear(x + gx * (offset - 0.25f), y + gy * (offset - 0.25f));
				profile[k + HALF] += step / contrast;
			}
			edges++;
		}
	}

	double weight = 0.0, mean = 0.0, second = 0.0;
	for (int k = -HALF; k <= HALF; k++) {
		double step = std::max(profile[k + HALF], 0.0);
		weight += step;
		mean += step * k * 0.5;
		second += step * k * k * 0.25;
	}
	if (edges == 0 || weight <= 0.0) return match;
	mean /= weight;
	std::vector<float> widths = { static_cast<float>(second / weight - mean * mean) };

	// A perfectly sharp edge still spreads over about one pixel once sampled and interpolated
	float variance = median(widths) - 0.65f;
	match.blur = variance > 0.04f ? std::sqrt(variance) : 0.0f;
	return match;
}

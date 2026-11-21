#include "lightFit.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <stb_image.h>

#include "scene/raycast.hpp"

using enginemath::Vec3;

namespace {
	constexpr double RIDGE = 1e-9;

	// Solves the n x n system in place (Gaussian elimination, partial pivoting). Returns false if singular
	bool solve(std::vector<double>& matrix, std::vector<double>& rhs, int n) {
		for (int col = 0; col < n; col++) {
			int pivot = col;
			for (int row = col + 1; row < n; row++) {
				if (std::abs(matrix[row * n + col]) > std::abs(matrix[pivot * n + col])) pivot = row;
			}
			if (std::abs(matrix[pivot * n + col]) < 1e-15) return false;
			if (pivot != col) {
				for (int k = 0; k < n; k++) std::swap(matrix[col * n + k], matrix[pivot * n + k]);
				std::swap(rhs[col], rhs[pivot]);
			}
			for (int row = col + 1; row < n; row++) {
				double factor = matrix[row * n + col] / matrix[col * n + col];
				for (int k = col; k < n; k++) matrix[row * n + k] -= factor * matrix[col * n + k];
				rhs[row] -= factor * rhs[col];
			}
		}
		for (int row = n - 1; row >= 0; row--) {
			for (int k = row + 1; k < n; k++) rhs[row] -= matrix[row * n + k] * rhs[k];
			rhs[row] /= matrix[row * n + row];
		}
		return true;
	}

	double lightTerm(const std::vector<recast::Proxy>& proxies, const recast::LightingSample& sample, const Vec3& lightPosition) {
		Vec3 toLight = lightPosition - sample.point;
		double distanceSquared = std::max(static_cast<double>(toLight.dot(toLight)), 1e-6);
		double distance = std::sqrt(distanceSquared);
		Vec3 direction = toLight / static_cast<float>(distance);

		double cosine = sample.normal.dot(direction);
		if (cosine <= 0.0) return 0.0;

		recast::RayHit blocker = recast::raycastProxies(proxies, sample.point + sample.normal * 1e-3f, direction);
		if (blocker.valid() && blocker.t < distance - 1e-3) return 0.0;

		return cosine / distanceSquared;
	}

	struct ProxySolution {
		int sampleCount = 0;
		std::vector<double> coefficients[3];
		std::vector<double> lightWeight;
	};

	std::vector<ProxySolution> solvePerProxy(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples,
											 const std::vector<Vec3>& lights, double& residual) {
		int unknowns = 1 + static_cast<int>(lights.size());
		size_t proxyCount = scene.proxies.size();

		std::vector<std::vector<double>> normal(proxyCount, std::vector<double>(unknowns * unknowns, 0.0));
		std::vector<std::vector<double>> rhs[3];
		for (auto& channel : rhs) channel.assign(proxyCount, std::vector<double>(unknowns, 0.0));
		std::vector<ProxySolution> solutions(proxyCount);
		for (auto& solution : solutions) solution.lightWeight.assign(lights.size(), 0.0);

		std::vector<std::vector<double>> features(samples.size(), std::vector<double>(unknowns, 1.0));
		for (size_t s = 0; s < samples.size(); s++) {
			const recast::LightingSample& sample = samples[s];
			ProxySolution& solution = solutions[sample.proxy];
			solution.sampleCount++;

			std::vector<double>& f = features[s];
			for (size_t l = 0; l < lights.size(); l++) {
				f[1 + l] = lightTerm(scene.proxies, sample, lights[l]);
				solution.lightWeight[l] += f[1 + l] * f[1 + l];
			}

			std::vector<double>& n = normal[sample.proxy];
			for (int i = 0; i < unknowns; i++) {
				for (int j = 0; j < unknowns; j++) n[i * unknowns + j] += f[i] * f[j];
				for (int c = 0; c < 3; c++) rhs[c][sample.proxy][i] += f[i] * sample.color[c];
			}
		}

		for (size_t p = 0; p < proxyCount; p++) {
			for (int c = 0; c < 3; c++) {
				std::vector<double> matrix = normal[p];
				for (int i = 0; i < unknowns; i++) matrix[i * unknowns + i] += RIDGE * (1.0 + matrix[i * unknowns + i]);
				std::vector<double> solution = rhs[c][p];
				if (!solve(matrix, solution, unknowns)) solution.assign(unknowns, 0.0);
				solutions[p].coefficients[c] = solution;
			}
		}

		residual = 0.0;
		for (size_t s = 0; s < samples.size(); s++) {
			const ProxySolution& solution = solutions[samples[s].proxy];
			for (int c = 0; c < 3; c++) {
				double predicted = 0.0;
				for (int i = 0; i < unknowns; i++) predicted += solution.coefficients[c][i] * features[s][i];
				double error = samples[s].color[c] - predicted;
				residual += error * error;
			}
		}
		return solutions;
	}

	double residualFor(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples, const std::vector<Vec3>& lights) {
		double residual = 0.0;
		solvePerProxy(scene, samples, lights, residual);
		return residual;
	}
}

Vec3 recast::LinearImage::at(int x, int y) const {
	size_t index = (static_cast<size_t>(y) * width + x) * 3;
	return Vec3(rgb[index], rgb[index + 1], rgb[index + 2]);
}

recast::LinearImage recast::loadLinearImage(const std::string& path) {
	int width = 0, height = 0, channels = 0;
	float* pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 3);
	if (!pixels) {
		throw std::runtime_error("ERROR: lightFit.cpp -> loadLinearImage(): " + path + " (" + stbi_failure_reason() + ")");
	}
	LinearImage image;
	image.width = width;
	image.height = height;
	image.rgb.assign(pixels, pixels + static_cast<size_t>(width) * height * 3);
	stbi_image_free(pixels);
	return image;
}

std::vector<recast::LightingSample> recast::collectLightingSamples(const SceneFile& scene, const LinearImage& photo, int stride) {
	const SceneCamera& camera = scene.camera;
	float scaleX = static_cast<float>(photo.width) / static_cast<float>(scene.image_size[0]);
	float scaleY = static_cast<float>(photo.height) / static_cast<float>(scene.image_size[1]);
	Vec3 origin(camera.position[0], camera.position[1], camera.position[2]);

	float maxValue = 0.0f;
	for (float value : photo.rgb) maxValue = std::max(maxValue, value);
	bool lowDynamicRange = maxValue <= 1.001f;

	std::vector<LightingSample> samples;
	for (int y = stride / 2; y < photo.height; y += stride) {
		for (int x = stride / 2; x < photo.width; x += stride) {
			Vec3 color = photo.at(x, y);
			if (lowDynamicRange && std::max({ color.x, color.y, color.z }) >= 0.99f) continue;

			float u = (x + 0.5f) / scaleX, v = (y + 0.5f) / scaleY;
			Vec3 local((u - camera.cx) / camera.fx, -(v - camera.cy) / camera.fy, -1.0f);
			Vec3 direction;
			for (int i = 0; i < 3; i++) {
				direction[i] = camera.rotation[i][0] * local.x + camera.rotation[i][1] * local.y + camera.rotation[i][2] * local.z;
			}

			RayHit hit = raycastProxies(scene.proxies, origin, direction.normalized());
			if (!hit.valid()) continue;
			samples.push_back({ color, hit.point, hit.normal, hit.proxyIndex });
		}
	}
	return samples;
}

recast::LightingFit recast::fitLightIntensities(const SceneFile& scene, const std::vector<LightingSample>& samples,
												const std::vector<Vec3>& lightPositions) {
	LightingFit fit;
	std::vector<ProxySolution> solutions = solvePerProxy(scene, samples, lightPositions, fit.residual);

	double ambientSum[3] = {}, ambientWeight = 0.0;
	std::vector<std::array<double, 3>> lightSum(lightPositions.size(), { 0.0, 0.0, 0.0 });
	std::vector<double> lightWeight(lightPositions.size(), 0.0);

	for (size_t p = 0; p < solutions.size(); p++) {
		const ProxySolution& solution = solutions[p];
		if (solution.sampleCount == 0) continue;
		const Vec3f& albedo = scene.proxies[p].material.albedo;

		for (int c = 0; c < 3; c++) {
			ambientSum[c] += solution.sampleCount * solution.coefficients[c][0] / std::max(albedo[c], 1e-3f);
		}
		ambientWeight += solution.sampleCount;

		for (size_t l = 0; l < lightPositions.size(); l++) {
			double weight = solution.lightWeight[l];
			for (int c = 0; c < 3; c++) {
				lightSum[l][c] += weight * solution.coefficients[c][1 + l] / std::max(albedo[c], 1e-3f);
			}
			lightWeight[l] += weight;
		}
	}

	for (int c = 0; c < 3; c++) {
		fit.ambient[c] = static_cast<float>(std::max(0.0, ambientWeight > 0.0 ? ambientSum[c] / ambientWeight : 0.0));
	}
	for (size_t l = 0; l < lightPositions.size(); l++) {
		Vec3 intensity;
		for (int c = 0; c < 3; c++) {
			intensity[c] = static_cast<float>(std::max(0.0, lightWeight[l] > 0.0 ? lightSum[l][c] / lightWeight[l] : 0.0));
		}
		fit.intensities.push_back(intensity);
	}
	return fit;
}

recast::LightingFit recast::fitLighting(const SceneFile& scene, const std::vector<LightingSample>& samples, std::vector<Vec3>& positions) {
	if (samples.empty()) return {};

	if (positions.empty()) {
		Vec3 boundsMin(1e30f), boundsMax(-1e30f);
		for (const LightingSample& sample : samples) {
			for (int a = 0; a < 3; a++) {
				boundsMin[a] = std::min(boundsMin[a], sample.point[a]);
				boundsMax[a] = std::max(boundsMax[a], sample.point[a]);
			}
		}
		boundsMin.y = std::max(boundsMin.y, 0.0f) + 0.2f;
		boundsMax.y = std::max(boundsMax.y, boundsMin.y + 1.0f);

		constexpr int steps = 7;
		double best = 1e300;
		Vec3 bestPosition;
		for (int i = 0; i < steps; i++) {
			for (int j = 0; j < steps; j++) {
				for (int k = 0; k < steps; k++) {
					Vec3 candidate(boundsMin.x + (boundsMax.x - boundsMin.x) * (i + 0.5f) / steps,
								   boundsMin.y + (boundsMax.y - boundsMin.y) * (j + 0.5f) / steps,
								   boundsMin.z + (boundsMax.z - boundsMin.z) * (k + 0.5f) / steps);
					double residual = residualFor(scene, samples, { candidate });
					if (residual < best) {
						best = residual;
						bestPosition = candidate;
					}
				}
			}
		}
		positions.push_back(bestPosition);
	}

	double current = residualFor(scene, samples, positions);
	for (float step = 0.25f; step > 0.004f; step *= 0.5f) {
		bool improved = true;
		while (improved) {
			improved = false;
			for (size_t l = 0; l < positions.size(); l++) {
				for (int axis = 0; axis < 3; axis++) {
					for (float sign : { -1.0f, 1.0f }) {
						std::vector<Vec3> trial = positions;
						trial[l][axis] += sign * step;
						double residual = residualFor(scene, samples, trial);
						if (residual < current) {
							current = residual;
							positions = trial;
							improved = true;
						}
					}
				}
			}
		}
	}

	return fitLightIntensities(scene, samples, positions);
}

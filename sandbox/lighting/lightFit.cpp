#include "lightFit.hpp"

#if defined(_MSC_VER) && !defined(NDEBUG)
#pragma optimize("gt", on)
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <stb_image.h>

#include "scene/raycast.hpp"

using enginemath::Vec3;

namespace {
	constexpr double RIDGE = 1e-9;
	constexpr double PI = 3.14159265358979323846;

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

	Vec3 directionFromAngles(double azimuth, double elevation) {
		return Vec3(static_cast<float>(std::cos(elevation) * std::sin(azimuth)), static_cast<float>(std::sin(elevation)),
					static_cast<float>(-std::cos(elevation) * std::cos(azimuth)));
	}

	void anglesFromDirection(const Vec3& direction, double& azimuth, double& elevation) {
		elevation = std::asin(std::clamp(static_cast<double>(direction.y), -1.0, 1.0));
		azimuth = std::atan2(direction.x, -direction.z);
	}

	struct Occluder {
		bool box = false;
		Vec3 point, normal, min, max;
	};

	struct Occluders {
		std::vector<Occluder> list;

		explicit Occluders(const std::vector<recast::Proxy>& proxies) {
			for (const recast::Proxy& proxy : proxies) {
				Occluder occluder;
				occluder.box = proxy.type == "box";
				if (!occluder.box && proxy.type != "plane") continue;
				occluder.point = Vec3(proxy.point[0], proxy.point[1], proxy.point[2]);
				occluder.normal = Vec3(proxy.normal[0], proxy.normal[1], proxy.normal[2]).normalized();
				occluder.min = Vec3(proxy.min[0], proxy.min[1], proxy.min[2]);
				occluder.max = Vec3(proxy.max[0], proxy.max[1], proxy.max[2]);
				list.push_back(occluder);
			}
		}

		// Any hit with 0 < t < maxT
		bool blocked(const Vec3& origin, const Vec3& direction, float maxT, bool boxesOnly) const {
			for (const Occluder& o : list) {
				if (o.box) {
					float tNear = 0.0f, tFar = maxT;
					bool miss = false;
					for (int a = 0; a < 3 && !miss; a++) {
						if (std::abs(direction[a]) < 1e-9f) {
							miss = origin[a] < o.min[a] || origin[a] > o.max[a];
							continue;
						}
						float inverse = 1.0f / direction[a];
						float t0 = (o.min[a] - origin[a]) * inverse;
						float t1 = (o.max[a] - origin[a]) * inverse;
						if (t0 > t1) std::swap(t0, t1);
						tNear = std::max(tNear, t0);
						tFar = std::min(tFar, t1);
						miss = tNear > tFar;
					}
					if (!miss) return true;
				}
				else if (!boxesOnly) {
					float denominator = o.normal.dot(direction);
					if (std::abs(denominator) < 1e-6f) continue;
					float t = (o.point - origin).dot(o.normal) / denominator;
					if (t > 1e-5f && t < maxT) return true;
				}
			}
			return false;
		}
	};

	double lightTerm(const Occluders& occluders, const recast::LightingSample& sample, const recast::FitLight& light) {
		Vec3 origin = sample.point + sample.normal * 1e-3f;

		if (light.directional) {
			Vec3 toLight = -light.direction;
			double cosine = sample.normal.dot(toLight);
			if (cosine <= 0.0) return 0.0;
			if (occluders.blocked(origin, toLight, 1e30f, true)) return 0.0;
			return cosine;
		}

		Vec3 toLight = light.position - sample.point;
		double distanceSquared = std::max(static_cast<double>(toLight.dot(toLight)), 1e-6);
		double distance = std::sqrt(distanceSquared);
		Vec3 direction = toLight / static_cast<float>(distance);

		double cosine = sample.normal.dot(direction);
		if (cosine <= 0.0) return 0.0;

		if (occluders.blocked(origin, direction, static_cast<float>(distance - 1e-3), false)) return 0.0;

		return cosine / distanceSquared;
	}

	struct ProxySolution {
		int sampleCount = 0;
		std::vector<double> coefficients[3];
		std::vector<double> lightWeight;
		std::vector<bool> lightVaries; // false: the light is (nearly) constant here, so it can't be told apart from ambient
	};

	std::vector<ProxySolution> solvePerProxy(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples,
											 const std::vector<recast::FitLight>& lights, double& residual) {
		Occluders occluders(scene.proxies);
		int unknowns = 1 + static_cast<int>(lights.size());
		size_t proxyCount = scene.proxies.size();

		std::vector<std::vector<double>> normal(proxyCount, std::vector<double>(unknowns * unknowns, 0.0));
		std::vector<std::vector<double>> rhs[3];
		for (auto& channel : rhs) channel.assign(proxyCount, std::vector<double>(unknowns, 0.0));
		std::vector<ProxySolution> solutions(proxyCount);
		for (auto& solution : solutions) solution.lightWeight.assign(lights.size(), 0.0);

		std::vector<double> features(samples.size() * unknowns, 1.0);
		for (size_t s = 0; s < samples.size(); s++) {
			const recast::LightingSample& sample = samples[s];
			ProxySolution& solution = solutions[sample.proxy];
			solution.sampleCount++;

			double* f = &features[s * unknowns];
			for (size_t l = 0; l < lights.size(); l++) {
				f[1 + l] = lightTerm(occluders, sample, lights[l]);
				solution.lightWeight[l] += f[1 + l] * f[1 + l];
			}

			std::vector<double>& n = normal[sample.proxy];
			for (int i = 0; i < unknowns; i++) {
				for (int j = 0; j < unknowns; j++) n[i * unknowns + j] += f[i] * f[j];
				for (int c = 0; c < 3; c++) rhs[c][sample.proxy][i] += f[i] * sample.color[c];
			}
		}

		for (size_t p = 0; p < proxyCount; p++) {
			const std::vector<double>& n = normal[p];
			double count = std::max(n[0], 1.0);
			for (size_t l = 0; l < lights.size(); l++) {
				double mean = n[1 + l] / count;
				double variance = n[(1 + l) * unknowns + (1 + l)] / count - mean * mean;
				solutions[p].lightVaries.push_back(mean > 0.0 && variance > 0.01 * mean * mean);
			}

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
				for (int i = 0; i < unknowns; i++) predicted += solution.coefficients[c][i] * features[s * unknowns + i];
				double error = samples[s].color[c] - predicted;
				residual += error * error;
			}
		}
		return solutions;
	}

	double residualFor(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples,
					   const std::vector<recast::FitLight>& lights) {
		double residual = 0.0;
		solvePerProxy(scene, samples, lights, residual);
		return residual;
	}

	void sampleBounds(const std::vector<recast::LightingSample>& samples, Vec3& boundsMin, Vec3& boundsMax) {
		boundsMin = Vec3(1e30f);
		boundsMax = Vec3(-1e30f);
		for (const recast::LightingSample& sample : samples) {
			for (int a = 0; a < 3; a++) {
				boundsMin[a] = std::min(boundsMin[a], sample.point[a]);
				boundsMax[a] = std::max(boundsMax[a], sample.point[a]);
			}
		}
		boundsMin.y = std::max(boundsMin.y, 0.0f) + 0.2f;
		boundsMax.y = std::max(boundsMax.y, boundsMin.y + 1.0f);
	}

	// Best extra light of one kind on a coarse grid, with `fixed` lights held in place
	recast::FitLight gridSearch(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples,
								const std::vector<recast::FitLight>& fixed, bool directional, double& bestResidual) {
		std::vector<recast::FitLight> trial = fixed;
		trial.push_back({});
		recast::FitLight best;
		bestResidual = 1e300;

		if (directional) {
			for (int a = 0; a < 16; a++) {
				for (int e = 0; e < 6; e++) {
					recast::FitLight candidate;
					candidate.directional = true;
					candidate.direction = directionFromAngles(2.0 * PI * a / 16.0, -PI / 180.0 * (10.0 + 14.0 * e));
					trial.back() = candidate;
					double residual = residualFor(scene, samples, trial);
					if (residual < bestResidual) {
						bestResidual = residual;
						best = candidate;
					}
				}
			}
			return best;
		}

		Vec3 boundsMin, boundsMax;
		sampleBounds(samples, boundsMin, boundsMax);
		constexpr int steps = 6;
		for (int i = 0; i < steps; i++) {
			for (int j = 0; j < steps; j++) {
				for (int k = 0; k < steps; k++) {
					recast::FitLight candidate;
					candidate.position = Vec3(boundsMin.x + (boundsMax.x - boundsMin.x) * (i + 0.5f) / steps,
											  boundsMin.y + (boundsMax.y - boundsMin.y) * (j + 0.5f) / steps,
											  boundsMin.z + (boundsMax.z - boundsMin.z) * (k + 0.5f) / steps);
					trial.back() = candidate;
					double residual = residualFor(scene, samples, trial);
					if (residual < bestResidual) {
						bestResidual = residual;
						best = candidate;
					}
				}
			}
		}
		return best;
	}

	bool allIntensitiesPositive(const recast::LightingFit& fit) {
		for (const Vec3& intensity : fit.intensities) {
			if (intensity.x + intensity.y + intensity.z < 1e-3f) return false;
		}
		return true;
	}

	double refine(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples,
				  std::vector<recast::FitLight>& lights, float startStep) {
		double current = residualFor(scene, samples, lights);
		for (float step = startStep; step > 0.004f; step *= 0.5f) {
			bool improved = true;
			while (improved) {
				improved = false;
				for (size_t l = 0; l < lights.size(); l++) {
					for (int axis = 0; axis < 3; axis++) {
						if (lights[l].directional && axis == 2) continue;
						for (float sign : { -1.0f, 1.0f }) {
							std::vector<recast::FitLight> trial = lights;
							if (trial[l].directional) {
								double azimuth, elevation;
								anglesFromDirection(trial[l].direction, azimuth, elevation);
								(axis == 0 ? azimuth : elevation) += sign * step * 0.5;
								elevation = std::clamp(elevation, -PI / 2.0 + 0.01, PI / 2.0 - 0.01);
								trial[l].direction = directionFromAngles(azimuth, elevation);
							}
							else {
								trial[l].position[axis] += sign * step;
							}

							double residual = residualFor(scene, samples, trial);
							if (residual < current) {
								current = residual;
								lights = trial;
								improved = true;
							}
						}
					}
				}
			}
		}
		return current;
	}

	Vec3 cameraRay(const recast::SceneCamera& camera, float u, float v) {
		Vec3 local((u - camera.cx) / camera.fx, -(v - camera.cy) / camera.fy, -1.0f);
		Vec3 direction;
		for (int i = 0; i < 3; i++) {
			direction[i] = camera.rotation[i][0] * local.x + camera.rotation[i][1] * local.y + camera.rotation[i][2] * local.z;
		}
		return direction.normalized();
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

			RayHit hit = raycastProxies(scene.proxies, origin, cameraRay(camera, (x + 0.5f) / scaleX, (y + 0.5f) / scaleY));
			if (!hit.valid()) continue;
			samples.push_back({ color, hit.point, hit.normal, hit.proxyIndex });
		}
	}
	return samples;
}

recast::LightingFit recast::fitLightIntensities(const SceneFile& scene, const std::vector<LightingSample>& samples,
												const std::vector<FitLight>& lights) {
	LightingFit fit;
	std::vector<ProxySolution> solutions = solvePerProxy(scene, samples, lights, fit.residual);

	double ambientSum[3] = {}, ambientWeight = 0.0;
	std::vector<std::array<double, 3>> lightSum(lights.size(), { 0.0, 0.0, 0.0 });
	std::vector<double> lightWeight(lights.size(), 0.0);

	bool anyIdentifiable = false;
	for (const ProxySolution& solution : solutions) {
		bool identifiable = solution.sampleCount > 0;
		for (bool varies : solution.lightVaries) identifiable = identifiable && varies;
		anyIdentifiable |= identifiable;
	}

	for (size_t p = 0; p < solutions.size(); p++) {
		const ProxySolution& solution = solutions[p];
		if (solution.sampleCount == 0) continue;
		const Vec3f& albedo = scene.proxies[p].material.albedo;

		bool identifiable = true;
		for (bool varies : solution.lightVaries) identifiable = identifiable && varies;
		if (identifiable || !anyIdentifiable) {
			for (int c = 0; c < 3; c++) {
				ambientSum[c] += solution.sampleCount * solution.coefficients[c][0] / std::max(albedo[c], 1e-3f);
			}
			ambientWeight += solution.sampleCount;
		}

		for (size_t l = 0; l < lights.size(); l++) {
			if (!solution.lightVaries[l] && anyIdentifiable) continue;
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
	for (size_t l = 0; l < lights.size(); l++) {
		Vec3 intensity;
		for (int c = 0; c < 3; c++) {
			intensity[c] = static_cast<float>(std::max(0.0, lightWeight[l] > 0.0 ? lightSum[l][c] / lightWeight[l] : 0.0));
		}
		fit.intensities.push_back(intensity);
	}
	return fit;
}

std::vector<Vec3> recast::detectBrightSpots(const SceneFile& scene, const LinearImage& photo, int maxSpots) {
	constexpr int cell = 4;
	int gridWidth = photo.width / cell, gridHeight = photo.height / cell;
	std::vector<float> luminance(static_cast<size_t>(gridWidth) * gridHeight, 0.0f);
	float brightest = 0.0f;

	for (int gy = 0; gy < gridHeight; gy++) {
		for (int gx = 0; gx < gridWidth; gx++) {
			float sum = 0.0f;
			for (int y = 0; y < cell; y++) {
				for (int x = 0; x < cell; x++) {
					Vec3 c = photo.at(gx * cell + x, gy * cell + y);
					sum += 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
				}
			}
			luminance[gy * gridWidth + gx] = sum / (cell * cell);
			brightest = std::max(brightest, luminance[gy * gridWidth + gx]);
		}
	}

	float threshold = brightest * 0.9f;
	std::vector<int> label(luminance.size(), -1);
	struct Blob { double energy = 0.0, x = 0.0, y = 0.0; int cells = 0; };
	std::vector<Blob> blobs;

	for (int start = 0; start < static_cast<int>(luminance.size()); start++) {
		if (label[start] >= 0 || luminance[start] < threshold) continue;
		Blob blob;
		std::vector<int> stack = { start };
		label[start] = static_cast<int>(blobs.size());
		while (!stack.empty()) {
			int index = stack.back();
			stack.pop_back();
			int gx = index % gridWidth, gy = index / gridWidth;
			blob.energy += luminance[index];
			blob.x += (gx + 0.5) * cell * luminance[index];
			blob.y += (gy + 0.5) * cell * luminance[index];
			blob.cells++;
			const int offsets[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
			for (const auto& offset : offsets) {
				int nx = gx + offset[0], ny = gy + offset[1];
				if (nx < 0 || ny < 0 || nx >= gridWidth || ny >= gridHeight) continue;
				int neighbour = ny * gridWidth + nx;
				if (label[neighbour] >= 0 || luminance[neighbour] < threshold) continue;
				label[neighbour] = label[start];
				stack.push_back(neighbour);
			}
		}
		if (blob.cells >= 3) blobs.push_back(blob);
	}

	std::sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) { return a.energy > b.energy; });

	const SceneCamera& camera = scene.camera;
	float scaleX = static_cast<float>(photo.width) / static_cast<float>(scene.image_size[0]);
	float scaleY = static_cast<float>(photo.height) / static_cast<float>(scene.image_size[1]);
	Vec3 origin(camera.position[0], camera.position[1], camera.position[2]);

	std::vector<Vec3> seeds;
	for (const Blob& blob : blobs) {
		if (static_cast<int>(seeds.size()) >= maxSpots) break;
		float u = static_cast<float>(blob.x / blob.energy) / scaleX;
		float v = static_cast<float>(blob.y / blob.energy) / scaleY;
		RayHit hit = raycastProxies(scene.proxies, origin, cameraRay(camera, u, v));
		if (hit.valid()) seeds.push_back(hit.point + hit.normal * 0.3f);
	}
	return seeds;
}

recast::LightingFit recast::autoFitLighting(const SceneFile& scene, const LinearImage& photo, const std::vector<LightingSample>& samples,
											std::vector<FitLight>& lights, int maxLights) {
	lights.clear();
	if (samples.empty()) return {};

	std::vector<LightingSample> coarse;
	for (size_t s = 0; s < samples.size(); s += 4) coarse.push_back(samples[s]);

	std::vector<Vec3> seeds = detectBrightSpots(scene, photo);
	double current = 1e300;

	while (static_cast<int>(lights.size()) < maxLights) {
		std::vector<std::vector<FitLight>> candidates;

		for (const Vec3& seed : seeds) {
			std::vector<FitLight> candidate = lights;
			candidate.push_back({ .directional = false, .position = seed });
			candidates.push_back(candidate);
		}
		for (bool directional : { false, true }) {
			double residual = 0.0;
			std::vector<FitLight> candidate = lights;
			candidate.push_back(gridSearch(scene, coarse, lights, directional, residual));
			candidates.push_back(candidate);
		}

		double bestResidual = 1e300;
		std::vector<FitLight> best;
		for (std::vector<FitLight>& candidate : candidates) {
			double residual = refine(scene, coarse, candidate, 0.25f);
			if (residual < bestResidual && allIntensitiesPositive(fitLightIntensities(scene, coarse, candidate))) {
				bestResidual = residual;
				best = candidate;
			}
		}

		if (best.empty() || bestResidual > current * 0.95) break;
		current = bestResidual;
		lights = best;
	}

	refine(scene, samples, lights, 0.03f);
	return fitLightIntensities(scene, samples, lights);
}

std::vector<recast::FitLight> recast::fitLightsFromScene(const SceneFile& scene) {
	std::vector<FitLight> lights;
	for (const SceneLight& light : scene.lights) {
		if (light.type == "point") {
			lights.push_back({ .directional = false, .position = Vec3(light.position[0], light.position[1], light.position[2]) });
		}
		else if (light.type == "directional") {
			Vec3 direction(light.direction[0], light.direction[1], light.direction[2]);
			lights.push_back({ .directional = true, .direction = direction.normalized() });
		}
	}
	return lights;
}

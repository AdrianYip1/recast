#include "lightFit.hpp"

#if defined(_MSC_VER) && !defined(NDEBUG)
#pragma optimize("gt", on)
#endif

#include <algorithm>
#include <map>
#include <tuple>
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

		// windows: openings in their wall (centre, unit along the wall, half width, half height, facing)
		struct Window { Vec3 centre, along; float halfWidth = 0.0f, halfHeight = 0.0f; Vec3 normal; };
		std::vector<Window> windows;

		explicit Occluders(const std::vector<recast::Proxy>& proxies, const std::vector<recast::SceneLight>* lights = nullptr) {
			if (lights) {
				for (const recast::SceneLight& light : *lights) {
					if (light.type != "area") continue;
					Vec3 c[4];
					for (int k = 0; k < 4; k++) c[k] = Vec3(light.corners[k][0], light.corners[k][1], light.corners[k][2]);
					Window window;
					window.centre = (c[0] + c[1] + c[2] + c[3]) * 0.25f;
					Vec3 across = (c[1] - c[0]) * 0.5f;
					window.halfWidth = std::sqrt(across.dot(across));
					if (window.halfWidth < 1e-4f) continue;
					window.along = across / window.halfWidth;
					window.halfHeight = 0.5f * (c[3].y - c[0].y);
					window.normal = window.along.cross(Vec3(0.0f, 1.0f, 0.0f)).normalized();
					windows.push_back(window);
				}
			}
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

		bool inWindow(const Vec3& position) const {
			for (const Window& w : windows) {
				Vec3 offset = position - w.centre;
				if (std::abs(offset.dot(w.normal)) > 0.15f) continue;
				if (std::abs(offset.dot(w.along)) <= w.halfWidth && std::abs(offset.y) <= w.halfHeight) return true;
			}
			return false;
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
					if (t > 1e-5f && t < maxT && !inWindow(origin + direction * t)) return true;
				}
			}
			return false;
		}
	};

	constexpr int SHAPE_SAMPLES = 32;

	// Evenly spread points: on the unit sphere (point lights) or on a cap of the given angle around +z (directional)
	Vec3 fibonacciPoint(int k, double capCosine) {
		double golden = PI * (3.0 - std::sqrt(5.0));
		double z = 1.0 - (1.0 - capCosine) * (k + 0.5) / SHAPE_SAMPLES;
		double r = std::sqrt(std::max(0.0, 1.0 - z * z));
		return Vec3(static_cast<float>(r * std::cos(golden * k)), static_cast<float>(r * std::sin(golden * k)), static_cast<float>(z));
	}

	Vec3 aroundAxis(const Vec3& local, const Vec3& axis) {
		Vec3 tangent = (std::abs(axis.x) > 0.5f ? axis.cross(Vec3(0.0f, 1.0f, 0.0f)) : axis.cross(Vec3(1.0f, 0.0f, 0.0f))).normalized();
		Vec3 bitangent = axis.cross(tangent);
		return tangent * local.x + bitangent * local.y + axis * local.z;
	}

	// Fraction of the light's shape visible from the sample
	double visibility(const Occluders& occluders, const Vec3& origin, const recast::FitLight& light) {
		int count = light.size > 0.0f ? SHAPE_SAMPLES : 1;
		int visible = 0;
		for (int k = 0; k < count; k++) {
			if (light.directional) {
				Vec3 direction = light.size > 0.0f ? aroundAxis(fibonacciPoint(k, std::cos(light.size)), -light.direction) : -light.direction;
				visible += occluders.blocked(origin, direction, 1e30f, true) ? 0 : 1;
			}
			else {
				Vec3 target = light.size > 0.0f ? light.position + fibonacciPoint(k, -1.0) * light.size : light.position;
				Vec3 toLight = target - origin;
				float distance = std::sqrt(toLight.dot(toLight));
				visible += occluders.blocked(origin, toLight / distance, distance - 1e-3f, false) ? 0 : 1;
			}
		}
		return static_cast<double>(visible) / count;
	}

	double lightTerm(const Occluders& occluders, const recast::LightingSample& sample, const recast::FitLight& light) {
		Vec3 origin = sample.point + sample.normal * 1e-3f;

		if (light.area) {
			// irradiance per unit radiance: a 4x4 grid of points on the rectangle, each cos * cos / d^2 * its share of the area
			Vec3 up(0.0f, light.halfHeight, 0.0f);
			Vec3 facing = light.halfWidth.cross(Vec3(0.0f, 1.0f, 0.0f)).normalized();
			double cell = 4.0 * std::sqrt(light.halfWidth.dot(light.halfWidth)) * light.halfHeight / 16.0;
			double sum = 0.0;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					Vec3 target = light.position + light.halfWidth * ((i + 0.5f) / 2.0f - 1.0f) + up * ((j + 0.5f) / 2.0f - 1.0f);
					Vec3 toLight = target - origin;
					float distanceSquared = std::max(toLight.dot(toLight), 1e-6f);
					float distance = std::sqrt(distanceSquared);
					Vec3 direction = toLight / distance;
					double cosSurface = sample.normal.dot(direction), cosLight = facing.dot(-direction);
					if (cosSurface <= 0.0 || cosLight <= 0.0) continue;
					if (occluders.blocked(origin, direction, distance - 1e-3f, false)) continue;
					sum += cosSurface * cosLight / distanceSquared;
				}
			}
			return sum * cell;
		}

		if (light.directional) {
			double cosine = sample.normal.dot(-light.direction);
			if (cosine <= 0.0) return 0.0;
			return cosine * visibility(occluders, origin, light);
		}

		Vec3 toLight = light.position - sample.point;
		double distanceSquared = std::max(static_cast<double>(toLight.dot(toLight)), 1e-6);
		double cosine = sample.normal.dot(toLight / static_cast<float>(std::sqrt(distanceSquared)));
		if (cosine <= 0.0) return 0.0;
		return cosine / distanceSquared * visibility(occluders, origin, light);
	}

	struct ProxySolution {
		int sampleCount = 0;
		std::vector<double> coefficients[3];
		std::vector<double> lightWeight;
		std::vector<bool> lightVaries; // false: the light is (nearly) constant here, so it can't be told apart from ambient
	};

	// Terms of fixed lights per sample set, kept for one fit (keyed by the sample set and the light's shape; cleared at
	// the start and end of each fit, since a later sample set can reuse the same memory)
	using TermKey = std::tuple<const void*, size_t, bool, float, float, float, float, float, float, float, float>;
	thread_local std::map<TermKey, std::vector<double>> fixedTermCache;

	const std::vector<double>& fixedLightTerms(const Occluders& occluders, const std::vector<recast::LightingSample>& samples,
											   const recast::FitLight& light) {
		auto& cache = fixedTermCache;
		TermKey key{ samples.data(), samples.size(), light.area, light.position.x, light.position.y, light.position.z,
				 light.halfWidth.x, light.halfWidth.z, light.halfHeight, light.size, light.directional ? 1.0f : 0.0f };
		auto found = cache.find(key);
		if (found != cache.end()) return found->second;
		if (cache.size() > 64) cache.clear();
		std::vector<double> terms(samples.size());
		for (size_t s = 0; s < samples.size(); s++) terms[s] = lightTerm(occluders, samples[s], light);
		return cache.emplace(key, std::move(terms)).first->second;
	}

	std::vector<ProxySolution> solvePerProxy(const recast::SceneFile& scene, const std::vector<recast::LightingSample>& samples,
											 const std::vector<recast::FitLight>& lights, double& residual) {
		Occluders occluders(scene.proxies, &scene.lights);
		int unknowns = 1 + static_cast<int>(lights.size());
		size_t proxyCount = scene.proxies.size();

		std::vector<std::vector<double>> normal(proxyCount, std::vector<double>(unknowns * unknowns, 0.0));
		std::vector<std::vector<double>> rhs[3];
		for (auto& channel : rhs) channel.assign(proxyCount, std::vector<double>(unknowns, 0.0));
		std::vector<ProxySolution> solutions(proxyCount);
		for (auto& solution : solutions) solution.lightWeight.assign(lights.size(), 0.0);

		// Lights found in the photo (fixed) never move during a fit, so their terms for these samples are worked out once
		std::vector<const std::vector<double>*> cached(lights.size(), nullptr);
		for (size_t l = 0; l < lights.size(); l++) {
			if (lights[l].fixed) cached[l] = &fixedLightTerms(occluders, samples, lights[l]);
		}

		std::vector<double> features(samples.size() * unknowns, 1.0);
		for (size_t s = 0; s < samples.size(); s++) {
			const recast::LightingSample& sample = samples[s];
			ProxySolution& solution = solutions[sample.proxy];
			solution.sampleCount++;

			double* f = &features[s * unknowns];
			for (size_t l = 0; l < lights.size(); l++) {
				f[1 + l] = cached[l] ? (*cached[l])[s] : lightTerm(occluders, sample, lights[l]);
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
					if (lights[l].fixed) continue;
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

namespace {
	struct Blob {
		double energy = 0.0, x = 0.0, y = 0.0; // x, y: centroid in photo pixels (energy weighted, until divided)
		int cells = 0;
		float peak = 0.0f;
	};

	// Connected regions of the brightest pixels (luminance >= relative * the brightest), brightest first
	std::vector<Blob> brightBlobs(const recast::LinearImage& photo, float relative, int& cellCount) {
		constexpr int cell = 4;
		int gridWidth = photo.width / cell, gridHeight = photo.height / cell;
		cellCount = gridWidth * gridHeight;
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

		float threshold = brightest * relative;
		std::vector<int> label(luminance.size(), -1);
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
				blob.peak = std::max(blob.peak, luminance[index]);
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
			if (blob.cells >= 3) {
				blob.x /= blob.energy;
				blob.y /= blob.energy;
				blobs.push_back(blob);
			}
		}
		std::sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) { return a.energy > b.energy; });
		return blobs;
	}
}

std::vector<Vec3> recast::detectBrightSpots(const SceneFile& scene, const LinearImage& photo, int maxSpots) {
	int cellCount = 0;
	std::vector<Blob> blobs = brightBlobs(photo, 0.9f, cellCount);
	const SceneCamera& camera = scene.camera;
	float scaleX = static_cast<float>(photo.width) / static_cast<float>(scene.image_size[0]);
	float scaleY = static_cast<float>(photo.height) / static_cast<float>(scene.image_size[1]);
	Vec3 origin(camera.position[0], camera.position[1], camera.position[2]);

	std::vector<Vec3> seeds;
	for (const Blob& blob : blobs) {
		if (static_cast<int>(seeds.size()) >= maxSpots) break;
		Vec3 direction = cameraRay(camera, static_cast<float>(blob.x) / scaleX, static_cast<float>(blob.y) / scaleY);
		RayHit hit = raycastProxies(scene.proxies, origin, direction);
		// a window or lamp with no wall behind it in the proxies still gives a seed, a few metres out along its ray
		seeds.push_back(hit.valid() ? hit.point + hit.normal * 0.3f : origin + direction * 3.0f);
	}
	return seeds;
}

// The sun, where the photo shows it: a small, clipped-bright region above the horizon whose ray reaches no furniture
// (it's seen through a window or in open sky). Returns the way its light travels, one per candidate
std::vector<Vec3> recast::sunDirections(const SceneFile& scene, const LinearImage& photo) {
	int cellCount = 0;
	std::vector<Blob> blobs = brightBlobs(photo, 0.97f, cellCount);
	const SceneCamera& camera = scene.camera;
	float scaleX = static_cast<float>(photo.width) / static_cast<float>(scene.image_size[0]);
	float scaleY = static_cast<float>(photo.height) / static_cast<float>(scene.image_size[1]);
	Vec3 origin(camera.position[0], camera.position[1], camera.position[2]);

	std::vector<Vec3> suns;
	for (const Blob& blob : blobs) {
		if (suns.size() >= 2) break;
		if (blob.cells > 0.015 * cellCount) continue; // a large bright area is sky or a window, not the sun's disc
		Vec3 direction = cameraRay(camera, static_cast<float>(blob.x) / scaleX, static_cast<float>(blob.y) / scaleY);
		if (direction.y < -0.05f) continue;           // the sun is at or above the horizon (a low sun is near it)
		RayHit hit = raycastProxies(scene.proxies, origin, direction);
		// glare on furniture or the floor isn't the sun
		if (hit.valid() && (scene.proxies[hit.proxyIndex].type == "box" || hit.normal.y > 0.7f)) continue;
		suns.push_back(-direction);
	}
	return suns;
}

namespace recast {
// Each light's size (bulb radius / sun angular radius) from how soft the photo's shadow edges are
static void fitLightSizes(const SceneFile& scene, const std::vector<LightingSample>& samples, std::vector<FitLight>& lights) {
	// Light sizes only change samples near shadow edges, so search on those plus a thin spread of the rest
	// (which keeps every surface's least squares solve stable)
	Occluders occluders(scene.proxies, &scene.lights);
	std::vector<LightingSample> sizeSamples;
	for (size_t s = 0; s < samples.size(); s++) {
		bool edge = false;
		Vec3 origin = samples[s].point + samples[s].normal * 1e-3f;
		for (const FitLight& light : lights) {
			FitLight widest = light;
			widest.size = light.directional ? enginemath::toRad(8.0f) : 0.35f;
			double visible = visibility(occluders, origin, widest);
			edge = edge || (visible > 0.0 && visible < 1.0) || visible != visibility(occluders, origin, light);
		}
		if (edge || s % 8 == 0) sizeSamples.push_back(samples[s]);
	}

	for (FitLight& light : lights) {
		if (light.fixed) continue;
		std::vector<float> sizes = light.directional
			? std::vector<float>{ 0.0f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f }
			: std::vector<float>{ 0.0f, 0.02f, 0.05f, 0.1f, 0.2f, 0.35f };
		if (light.directional) {
			for (float& size : sizes) size = enginemath::toRad(size);
		}
		auto residualAt = [&](float size) {
			light.size = size;
			return residualFor(scene, sizeSamples, lights);
		};

		size_t best = 0;
		double bestResidual = 1e300;
		for (size_t i = 0; i < sizes.size(); i++) {
			double residual = residualAt(sizes[i]);
			if (residual < bestResidual) {
				bestResidual = residual;
				best = i;
			}
		}

		// Golden-section search between the neighbours of the best candidate
		double low = sizes[best > 0 ? best - 1 : 0];
		double high = sizes[std::min(best + 1, sizes.size() - 1)];
		const double ratio = (std::sqrt(5.0) - 1.0) * 0.5;
		double a = high - ratio * (high - low), b = low + ratio * (high - low);
		double residualA = residualAt(static_cast<float>(a)), residualB = residualAt(static_cast<float>(b));
		for (int iteration = 0; iteration < 8; iteration++) {
			if (residualA < residualB) {
				high = b; b = a; residualB = residualA;
				a = high - ratio * (high - low);
				residualA = residualAt(static_cast<float>(a));
			}
			else {
				low = a; a = b; residualA = residualB;
				b = low + ratio * (high - low);
				residualB = residualAt(static_cast<float>(b));
			}
		}
		double refined = residualA < residualB ? a : b;
		light.size = std::min(residualA, residualB) < bestResidual ? static_cast<float>(refined) : sizes[best];
		bestResidual = std::min({ bestResidual, residualA, residualB });

		// Sharp unless softening clearly explains the photo better
		float chosen = light.size;
		light.size = residualAt(0.0f) <= bestResidual * 1.002 ? 0.0f : chosen;
	}
}
} // namespace recast

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
	fitLightSizes(scene, samples, lights);
	return fitLightIntensities(scene, samples, lights);
}

std::vector<recast::FitLight> recast::fitLightsFromScene(const SceneFile& scene) {
	std::vector<FitLight> lights;
	for (const SceneLight& light : scene.lights) {
		if (light.type == "point") {
			lights.push_back({ .directional = false, .position = Vec3(light.position[0], light.position[1], light.position[2]), .size = light.radius });
		}
		else if (light.type == "directional") {
			Vec3 direction(light.direction[0], light.direction[1], light.direction[2]);
			lights.push_back({ .directional = true, .direction = direction.normalized(), .size = enginemath::toRad(light.angular_radius_deg) });
		}
	}
	return lights;
}

// The lights found in the photo (`lights` on input: windows, lamps) stay where they are and only get their strength.
// One more light (the directional sun from a 96-direction grid, or a point lamp when that isn't a clear win) is added
// when it explains the photo at least 2% better, each refined locally on at most ~6000 samples. Well under a second
// where the full multi-light search takes 15 s (2 min in a debug build); most of what no light explains is colour
// variation within the furniture
recast::LightingFit recast::fitMainLight(const SceneFile& scene, const std::vector<LightingSample>& samples,
										 std::vector<FitLight>& lights, const std::vector<Vec3>& suns) {
	fixedTermCache.clear();
	struct ClearAfter { ~ClearAfter() { fixedTermCache.clear(); } } clearAfter;
	std::vector<FitLight> found = lights;
	for (FitLight& light : found) light.fixed = true;
	lights.clear();
	if (samples.empty()) return {};
	std::vector<LightingSample> subset;
	size_t step = std::max<size_t>(1, samples.size() / 6000);
	for (size_t s = 0; s < samples.size(); s += step) subset.push_back(samples[s]);

	// a window or lamp that comes out with no strength isn't lighting anything we can see
	if (!found.empty()) {
		LightingFit check = fitLightIntensities(scene, subset, found);
		std::vector<FitLight> kept;
		for (size_t l = 0; l < found.size(); l++) {
			if (std::max({ check.intensities[l].x, check.intensities[l].y, check.intensities[l].z }) > 1e-4f) kept.push_back(found[l]);
		}
		found = kept;
	}
	lights = found;

	double baseline = residualFor(scene, subset, found);
	double bestResidual = baseline * 0.98;
	// The sun, where the photo shows it, is stronger evidence than the shading fit (the boxes can't explain the real
	// shadows of legs and frames), so when there is one that explains at least 2% of the shading it's the main light and
	// the searches are skipped. It keeps the
	// photo's side of the sky; its height is refined but held at least 12 degrees up, since a sun at the horizon lights
	// nothing and the camera's pitch is the likely error. Windows or lamps that come out negative next to it are dropped
	bool sunFromPhoto = false;
	for (const Vec3& sun : suns) {
		constexpr double lowest = -12.0 * PI / 180.0; // elevation of the way the light travels: negative = downwards
		auto holdUp = [&](FitLight& light) {
			double azimuth, elevation;
			anglesFromDirection(light.direction, azimuth, elevation);
			light.direction = directionFromAngles(azimuth, std::clamp(elevation, -PI / 2.0 + 0.01, lowest));
		};
		std::vector<FitLight> candidate = found;
		candidate.push_back({ .directional = true, .direction = sun.normalized() });
		holdUp(candidate.back());
		refine(scene, subset, candidate, 0.25f);
		holdUp(candidate.back());
		LightingFit check = fitLightIntensities(scene, subset, candidate);
		std::vector<FitLight> kept;
		for (size_t l = 0; l + 1 < candidate.size(); l++) {
			if (std::min({ check.intensities[l].x, check.intensities[l].y, check.intensities[l].z }) > 0.0f) kept.push_back(candidate[l]);
		}
		const Vec3& sunStrength = check.intensities.back();
		if (std::max({ sunStrength.x, sunStrength.y, sunStrength.z }) <= 0.0f) continue;
		kept.push_back(candidate.back());
		double residual = residualFor(scene, subset, kept);
		if (residual > baseline * 0.98) continue; // a bright patch of sky that lights nothing we can see: not the sun
		lights = kept;
		bestResidual = residual;
		sunFromPhoto = true;
		break;
	}

	for (bool directional : { true, false }) {
		if (sunFromPhoto) break;
		double residual = 0.0;
		std::vector<FitLight> candidate = found;
		candidate.push_back(gridSearch(scene, subset, found, directional, residual));
		refine(scene, subset, candidate, 0.25f);
		residual = refine(scene, subset, candidate, 0.03f);
		if (residual < bestResidual && allIntensitiesPositive(fitLightIntensities(scene, subset, candidate))) {
			bestResidual = residual;
			lights = candidate;
		}
		if (lights.size() > found.size() && bestResidual < baseline * 0.8) break; // a clear sun: no need to look for a lamp
	}
	if (lights.empty()) return fitLightIntensities(scene, samples, lights);
	fitLightSizes(scene, subset, lights);
	LightingFit fit = fitLightIntensities(scene, samples, lights);

	// When the per-surface solves can't separate ambient from the light (colour changes within a piece of furniture
	// look like shading), the ambient collapses to zero and every shadow would be black. Then measure it where no
	// light reaches (shadows, surfaces facing away), and with a single light refit its strength against that ambient
	Occluders occluders(scene.proxies, &scene.lights);
	std::vector<double> terms(subset.size(), 0.0);
	for (size_t i = 0; i < subset.size(); i++) {
		for (const FitLight& light : lights) terms[i] += lightTerm(occluders, subset[i], light);
	}
	for (int c = 0; c < 3; c++) {
		if (fit.ambient[c] > 1e-3f) continue;
		std::vector<double> unlit;
		for (size_t i = 0; i < subset.size(); i++) {
			const LightingSample& sample = subset[i];
			if (terms[i] <= 0.0) unlit.push_back(sample.color[c] / std::max(scene.proxies[sample.proxy].material.albedo[c], 1e-3f));
		}
		if (unlit.size() < 20) continue;
		std::nth_element(unlit.begin(), unlit.begin() + unlit.size() / 2, unlit.end());
		double ambient = unlit[unlit.size() / 2];
		fit.ambient[c] = static_cast<float>(ambient);
		if (lights.size() != 1) continue;
		double numerator = 0.0, denominator = 0.0;
		for (size_t i = 0; i < subset.size(); i++) {
			if (terms[i] <= 0.0) continue;
			double irradiance = subset[i].color[c] / std::max(scene.proxies[subset[i].proxy].material.albedo[c], 1e-3f);
			numerator += terms[i] * (irradiance - ambient);
			denominator += terms[i] * terms[i];
		}
		if (denominator > 0.0) fit.intensities[0][c] = static_cast<float>(std::max(0.0, numerator / denominator));
	}
	return fit;
}

std::vector<recast::FitLight> recast::emittersToFit(const SceneFile& scene) {
	std::vector<FitLight> found;
	for (const SceneLight& light : scene.lights) {
		if (light.type == "area") {
			Vec3 c[4];
			for (int k = 0; k < 4; k++) c[k] = Vec3(light.corners[k][0], light.corners[k][1], light.corners[k][2]);
			FitLight window;
			window.area = true;
			window.position = (c[0] + c[1] + c[2] + c[3]) * 0.25f;
			window.halfWidth = (c[1] - c[0]) * 0.5f;
			window.halfHeight = 0.5f * (c[3].y - c[0].y);
			window.fixed = true;
			found.push_back(window);
		}
		else if (light.type == "point" && light.intensity == Vec3f{ 0.0f, 0.0f, 0.0f }) {
			found.push_back({ .position = Vec3(light.position[0], light.position[1], light.position[2]), .size = light.radius, .fixed = true });
		}
	}
	return found;
}

recast::SceneLight recast::sceneLightFrom(const FitLight& light, const Vec3& strength) {
	SceneLight out;
	if (light.area) {
		out.type = "area";
		Vec3 up(0.0f, light.halfHeight, 0.0f);
		Vec3 corners[4] = { light.position - light.halfWidth - up, light.position + light.halfWidth - up,
							light.position + light.halfWidth + up, light.position - light.halfWidth + up };
		for (int k = 0; k < 4; k++) out.corners[k] = { corners[k].x, corners[k].y, corners[k].z };
		out.radiance = { strength.x, strength.y, strength.z };
		return out;
	}
	out.type = light.directional ? "directional" : "point";
	out.position = { light.position.x, light.position.y, light.position.z };
	out.direction = { light.direction.x, light.direction.y, light.direction.z };
	out.intensity = { strength.x, strength.y, strength.z };
	if (light.directional) out.angular_radius_deg = light.size * 180.0f / enginemath::PI;
	else out.radius = light.size;
	return out;
}

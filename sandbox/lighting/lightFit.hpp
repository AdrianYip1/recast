#pragma once

#include <string>
#include <vector>

#include <enginemath/enginemath.hpp>

#include "scene/sceneFile.hpp"

namespace recast {
	struct LinearImage {
		int width = 0;
		int height = 0;
		std::vector<float> rgb;

		enginemath::Vec3 at(int x, int y) const;
	};

	// .hdr loads as-is; 8-bit formats are converted to linear by stb's gamma 2.2
	LinearImage loadLinearImage(const std::string& path);

	struct LightingSample {
		enginemath::Vec3 color;
		enginemath::Vec3 point;
		enginemath::Vec3 normal;
		int proxy = -1;
	};

	// One sample per `stride` pixels that sees a proxy. Clipped pixels of 8-bit photos are skipped
	std::vector<LightingSample> collectLightingSamples(const SceneFile& scene, const LinearImage& photo, int stride);

	struct LightingFit {
		enginemath::Vec3 ambient;
		std::vector<enginemath::Vec3> intensities;
		double residual = 0.0;
	};

	// photo = albedo * (ambient + sum intensity * cos / d^2 * visibility), solved per proxy with its own albedo scale,
	// then converted to absolute values with the proxies' albedo from scene.json
	LightingFit fitLightIntensities(const SceneFile& scene, const std::vector<LightingSample>& samples,
									const std::vector<enginemath::Vec3>& lightPositions);

	// Photo brightness divided by the lighting model's prediction on the supporting surface around `center`
	// (a disk of `radius` at center.y), per channel. 1 = the model already matches the photo there.
	// Returns 1 if too little of the surface is visible
	enginemath::Vec3 localLightingCalibration(const SceneFile& scene, const LinearImage& photo, const enginemath::Vec3& center, float radius);

	// Refines light positions (and finds a first one by grid search if `positions` is empty), then fits intensities
	LightingFit fitLighting(const SceneFile& scene, const std::vector<LightingSample>& samples, std::vector<enginemath::Vec3>& positions);
} // namespace recast

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

	// Point: falls off with distance squared. Directional: parallel rays, no falloff, `direction` is the way the light travels.
	// Directional lights are only shadowed by boxes: planes are assumed open towards a distant source (sun through an opening)
	// size: point = radius in metres, directional = angular radius in radians (soft shadow edges)
	struct FitLight {
		bool directional = false;
		enginemath::Vec3 position;
		enginemath::Vec3 direction = enginemath::Vec3(0.0f, -1.0f, 0.0f);
		float size = 0.0f;
	};

	struct LightingFit {
		enginemath::Vec3 ambient;
		std::vector<enginemath::Vec3> intensities;
		double residual = 0.0;
	};

	// photo = albedo * (ambient + sum intensity * lightTerm), solved per proxy with its own albedo scale,
	// then converted to absolute values with the proxies' albedo from scene.json
	LightingFit fitLightIntensities(const SceneFile& scene, const std::vector<LightingSample>& samples, const std::vector<FitLight>& lights);

	// Surface points under the brightest blobs of the photo (lamps, windows, hotspots), brightest first
	std::vector<enginemath::Vec3> detectBrightSpots(const SceneFile& scene, const LinearImage& photo, int maxSpots = 4);

	// Finds lights from scratch: tries point lights (bright spot seeds + grid search) and directional lights, keeps the
	// better model, and adds more lights (up to maxLights) only while each one lowers the residual by at least 5%.
	// Each light's size is then picked to match how soft the photo's shadow edges are
	// One main light plus ambient: directional (sun, window), or a point light (lamp) when that fits clearly better,
	// or ambient alone when neither helps. Under a second, where autoFitLighting's multi-light search takes much longer
	LightingFit fitMainLight(const SceneFile& scene, const std::vector<LightingSample>& samples, std::vector<FitLight>& lights);

	LightingFit autoFitLighting(const SceneFile& scene, const LinearImage& photo, const std::vector<LightingSample>& samples,
								std::vector<FitLight>& lights, int maxLights = 3);

	// Scene lights (point and directional) in fitting form
	std::vector<FitLight> fitLightsFromScene(const SceneFile& scene);
} // namespace recast

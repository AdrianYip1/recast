#pragma once

#include <limits>
#include <vector>

#include <enginemath/enginemath.hpp>

#include "sceneFile.hpp"

namespace recast {
	struct RayHit {
		float t = std::numeric_limits<float>::infinity();
		enginemath::Vec3 point;
		enginemath::Vec3 normal;
		int proxyIndex = -1;

		bool valid() const { return proxyIndex >= 0; }
	};

	// Nearest proxy hit. minNormalY filters surfaces (e.g. 0.7 = only floors and tops you can stand things on)
	RayHit raycastProxies(const std::vector<Proxy>& proxies, const enginemath::Vec3& origin, const enginemath::Vec3& direction,
						  float minNormalY = -2.0f);

	// Nearest hit on a "hull" proxy's triangles (t > 0), normal facing the ray
	bool rayHull(const Proxy& hull, const enginemath::Vec3& origin, const enginemath::Vec3& direction, float& t, enginemath::Vec3& normal);

	bool rayAabb(const enginemath::Vec3& origin, const enginemath::Vec3& direction,
				 const enginemath::Vec3& boxMin, const enginemath::Vec3& boxMax, float& t, enginemath::Vec3* normal = nullptr);
} // namespace recast

#pragma once

#include <vector>

#include <enginemath/enginemath.hpp>

#include "sceneFile.hpp"

namespace recast {
	enginemath::Mat4 objectMatrix(const ObjectTransform& transform);

	// World AABB of a mesh's local bounds under the object transform
	void objectWorldBounds(const ObjectTransform& transform, const enginemath::Vec3& localMin, const enginemath::Vec3& localMax,
						   enginemath::Vec3& worldMin, enginemath::Vec3& worldMax);

	// Pushes the object's footprint out of walls (vertical planes) and boxes it overlaps vertically, along the shortest way.
	// Floors and box tops it rests on don't count. Returns true if the position changed.
	bool resolveCollisions(const std::vector<Proxy>& proxies, ObjectTransform& transform,
						   const enginemath::Vec3& localMin, const enginemath::Vec3& localMax);
} // namespace recast

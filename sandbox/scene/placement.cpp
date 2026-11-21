#include "placement.hpp"

#include <algorithm>
#include <cmath>

using enginemath::Vec3;
using enginemath::Vec4;
using enginemath::Mat4;

Mat4 recast::objectMatrix(const ObjectTransform& t) {
	return Mat4::translationM(t.position[0], t.position[1], t.position[2])
		 * Mat4::rotateY(enginemath::toRad(t.rotation_y_deg))
		 * Mat4::scaleM(t.scale, t.scale, t.scale);
}

void recast::objectWorldBounds(const ObjectTransform& transform, const Vec3& localMin, const Vec3& localMax, Vec3& worldMin, Vec3& worldMax) {
	Mat4 matrix = objectMatrix(transform);
	worldMin = Vec3(1e30f);
	worldMax = Vec3(-1e30f);
	for (int i = 0; i < 8; i++) {
		Vec4 corner((i & 1) ? localMax.x : localMin.x, (i & 2) ? localMax.y : localMin.y, (i & 4) ? localMax.z : localMin.z, 1.0f);
		Vec4 world = matrix * corner;
		for (int a = 0; a < 3; a++) {
			worldMin[a] = std::min(worldMin[a], world[a]);
			worldMax[a] = std::max(worldMax[a], world[a]);
		}
	}
}

bool recast::resolveCollisions(const std::vector<Proxy>& proxies, ObjectTransform& transform, const Vec3& localMin, const Vec3& localMax) {
	constexpr float epsilon = 1e-3f;
	bool changed = false;

	for (int iteration = 0; iteration < 4; iteration++) {
		Vec3 worldMin, worldMax;
		objectWorldBounds(transform, localMin, localMax, worldMin, worldMax);
		Vec3 center = (worldMin + worldMax) * 0.5f;
		Vec3 half = (worldMax - worldMin) * 0.5f;
		bool moved = false;

		// The box the object stands on wins: boxes overlapping its footprint are loose detections (nothing is
		// inside a table top), so they don't push the object off it
		const Proxy* support = nullptr;
		for (const Proxy& proxy : proxies) {
			if (proxy.type == "box" && std::abs(proxy.max[1] - worldMin.y) < 0.02f && center.x >= proxy.min[0] && center.x <= proxy.max[0] &&
				center.z >= proxy.min[2] && center.z <= proxy.max[2]) {
				support = &proxy;
			}
		}
		auto overlapsSupport = [&](const Proxy& proxy) {
			return support && &proxy != support && proxy.min[0] < support->max[0] && proxy.max[0] > support->min[0] &&
				   proxy.min[2] < support->max[2] && proxy.max[2] > support->min[2];
		};

		for (const Proxy& proxy : proxies) {
			if (proxy.type == "plane") {
				Vec3 normal = Vec3(proxy.normal[0], proxy.normal[1], proxy.normal[2]).normalized();
				if (std::abs(normal.y) > 0.3f) continue;

				Vec3 point(proxy.point[0], proxy.point[1], proxy.point[2]);
				float extent = std::abs(normal.x) * half.x + std::abs(normal.z) * half.z;
				float penetration = extent - (center - point).dot(normal);
				if (penetration > 0.0f) {
					transform.position[0] += normal.x * penetration;
					transform.position[2] += normal.z * penetration;
					moved = true;
				}
			}
			else if (proxy.type == "box") {
				if (worldMin.y >= proxy.max[1] - epsilon || worldMax.y <= proxy.min[1] + epsilon) continue;
				if (overlapsSupport(proxy)) continue;

				float overlapX = std::min(worldMax.x, proxy.max[0]) - std::max(worldMin.x, proxy.min[0]);
				float overlapZ = std::min(worldMax.z, proxy.max[2]) - std::max(worldMin.z, proxy.min[2]);
				if (overlapX <= 0.0f || overlapZ <= 0.0f) continue;

				float boxCenterX = (proxy.min[0] + proxy.max[0]) * 0.5f;
				float boxCenterZ = (proxy.min[2] + proxy.max[2]) * 0.5f;
				if (overlapX < overlapZ) transform.position[0] += (center.x < boxCenterX ? -overlapX : overlapX);
				else transform.position[2] += (center.z < boxCenterZ ? -overlapZ : overlapZ);
				moved = true;
			}
		}

		changed |= moved;
		if (!moved) break;
	}

	return changed;
}

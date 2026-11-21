#include "raycast.hpp"

#include <algorithm>
#include <cmath>

using enginemath::Vec3;

static Vec3 toVec3(const recast::Vec3f& v) {
	return Vec3(v[0], v[1], v[2]);
}

bool recast::rayAabb(const Vec3& origin, const Vec3& direction, const Vec3& boxMin, const Vec3& boxMax, float& t, Vec3* normal) {
	float tNear = -std::numeric_limits<float>::infinity();
	float tFar = std::numeric_limits<float>::infinity();
	int nearAxis = 0;
	float nearSign = 1.0f;

	for (int a = 0; a < 3; a++) {
		if (std::abs(direction[a]) < 1e-9f) {
			if (origin[a] < boxMin[a] || origin[a] > boxMax[a]) return false;
			continue;
		}
		float t0 = (boxMin[a] - origin[a]) / direction[a];
		float t1 = (boxMax[a] - origin[a]) / direction[a];
		float sign = -1.0f;
		if (t0 > t1) {
			std::swap(t0, t1);
			sign = 1.0f;
		}
		if (t0 > tNear) {
			tNear = t0;
			nearAxis = a;
			nearSign = sign;
		}
		tFar = std::min(tFar, t1);
	}

	if (tNear > tFar || tFar < 0.0f) return false;
	t = tNear > 0.0f ? tNear : tFar;
	if (normal) {
		*normal = Vec3();
		(*normal)[nearAxis] = nearSign;
	}
	return true;
}

bool recast::rayHull(const Proxy& hull, const Vec3& origin, const Vec3& direction, float& t, Vec3& normal) {
	float boxT = 0.0f;
	if (!rayAabb(origin, direction, toVec3(hull.min), toVec3(hull.max), boxT)) return false;
	Vec3 anchor = toVec3(hull.point);
	t = std::numeric_limits<float>::infinity();
	for (const auto& face : hull.faces) {
		if (face[0] >= hull.vertices.size() || face[1] >= hull.vertices.size() || face[2] >= hull.vertices.size()) continue;
		Vec3 a = anchor + toVec3(hull.vertices[face[0]]);
		Vec3 e1 = anchor + toVec3(hull.vertices[face[1]]) - a, e2 = anchor + toVec3(hull.vertices[face[2]]) - a;
		Vec3 p = direction.cross(e2);
		float det = e1.dot(p);
		if (std::abs(det) < 1e-12f) continue;
		Vec3 s = origin - a;
		float u = s.dot(p) / det;
		if (u < 0.0f || u > 1.0f) continue;
		Vec3 q = s.cross(e1);
		float v = direction.dot(q) / det;
		if (v < 0.0f || u + v > 1.0f) continue;
		float hit = e2.dot(q) / det;
		if (hit > 1e-5f && hit < t) {
			t = hit;
			normal = e1.cross(e2).normalized();
		}
	}
	if (t == std::numeric_limits<float>::infinity()) return false;
	if (normal.dot(direction) > 0.0f) normal = -normal;
	return true;
}

recast::RayHit recast::raycastProxies(const std::vector<Proxy>& proxies, const Vec3& origin, const Vec3& direction, float minNormalY) {
	RayHit best;

	for (size_t i = 0; i < proxies.size(); i++) {
		const Proxy& proxy = proxies[i];
		float t = 0.0f;
		Vec3 normal;

		if (proxy.type == "plane") {
			normal = toVec3(proxy.normal).normalized();
			float denominator = normal.dot(direction);
			if (std::abs(denominator) < 1e-6f) continue;
			t = (toVec3(proxy.point) - origin).dot(normal) / denominator;
			if (t <= 0.0f) continue;
			if (denominator > 0.0f) normal = -normal;
		}
		else if (proxy.type == "box") {
			if (!rayAabb(origin, direction, toVec3(proxy.min), toVec3(proxy.max), t, &normal) || t <= 0.0f) continue;
		}
		else if (proxy.type == "hull") {
			if (!rayHull(proxy, origin, direction, t, normal)) continue;
		}
		else {
			continue;
		}

		if (normal.y < minNormalY || t >= best.t) continue;
		best.t = t;
		best.point = origin + direction * t;
		best.normal = normal;
		best.proxyIndex = static_cast<int>(i);
	}

	return best;
}

#include "proxyMesh.hpp"

#include <cmath>

using enginemath::Vec2;
using enginemath::Vec3;

static Vec3 toVec3(const recast::Vec3f& v) {
	return Vec3(v[0], v[1], v[2]);
}

static void addQuad(recast::MeshData& mesh, const Vec3& center, const Vec3& normal, const Vec3& axisU, const Vec3& axisV) {
	foto::u32 base = static_cast<foto::u32>(mesh.vertices.size());
	const Vec2 uvs[4] = { Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1) };
	const float signs[4][2] = { { -1, -1 }, { 1, -1 }, { 1, 1 }, { -1, 1 } };

	for (int i = 0; i < 4; i++) {
		mesh.vertices.push_back({ center + axisU * signs[i][0] + axisV * signs[i][1], normal, uvs[i] });
	}

	bool counterClockwise = axisU.cross(axisV).dot(normal) > 0.0f;
	if (counterClockwise) {
		mesh.indices.insert(mesh.indices.end(), { base, base + 1, base + 2, base + 2, base + 3, base });
	}
	else {
		mesh.indices.insert(mesh.indices.end(), { base, base + 2, base + 1, base + 2, base, base + 3 });
	}
}

recast::MeshData recast::buildProxyMesh(const Proxy& proxy, foto::f32 planeHalfSize) {
	MeshData mesh;

	if (proxy.type == "plane") {
		Vec3 normal = toVec3(proxy.normal).normalized();
		Vec3 reference = std::abs(normal.y) < 0.9f ? Vec3::up() : Vec3::right();
		Vec3 tangent = normal.cross(reference).normalized();
		Vec3 bitangent = normal.cross(tangent);
		addQuad(mesh, toVec3(proxy.point), normal, tangent * planeHalfSize, bitangent * planeHalfSize);
	}
	else if (proxy.type == "box") {
		Vec3 min = toVec3(proxy.min), max = toVec3(proxy.max);
		Vec3 center = (min + max) * 0.5f;
		Vec3 half = (max - min) * 0.5f;
		const Vec3 axes[3] = { Vec3::right(), Vec3::up(), Vec3(0, 0, 1) };

		for (int a = 0; a < 3; a++) {
			Vec3 u = axes[(a + 1) % 3] * half[(a + 1) % 3];
			Vec3 v = axes[(a + 2) % 3] * half[(a + 2) % 3];
			for (float side : { -1.0f, 1.0f }) {
				Vec3 normal = axes[a] * side;
				addQuad(mesh, center + normal * half[a], normal, u, v);
			}
		}
	}

	return mesh;
}

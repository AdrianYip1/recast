#pragma once

#include <vector>

#include <engine/rhi_vulkan/vkVertex.hpp>

#include "sceneFile.hpp"

namespace recast {
	struct MeshData {
		std::vector<foto::Vertex> vertices;
		std::vector<foto::u32> indices;
	};

	// Planes become a square of planeHalfSize around their point; boxes get 6 outward-facing quads
	MeshData buildProxyMesh(const Proxy& proxy, foto::f32 planeHalfSize = 10.0f);
} // namespace recast

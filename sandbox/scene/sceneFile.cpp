#include "sceneFile.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace recast {
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SceneCamera, fx, fy, cx, cy, rotation, position)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ScaleReference, kind, metres)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ObjectTransform, position, rotation_y_deg, scale)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SceneObject, path, transform)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ProxyMaterial, albedo, roughness)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Proxy, type, id, point, normal, min, max, material, vertices, faces)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SceneLight, type, position, direction, intensity, radius, angular_radius_deg, corners, radiance)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Lighting, ambient, sh_order, sh)
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SceneFile, version, photo, image_size, camera, scale_reference,
													object, proxies, lights, lighting)
}

std::string recast::SceneFile::resolve(const std::string& relativePath) const {
	if (relativePath.empty()) return relativePath;
	return (directory / std::filesystem::u8path(relativePath)).lexically_normal().string();
}

recast::SceneFile recast::loadSceneFile(const std::filesystem::path& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("ERROR: sceneFile.cpp -> loadSceneFile(): can't open " + path.string());
	}

	SceneFile scene;
	try {
		scene = nlohmann::json::parse(file).get<SceneFile>();
	}
	catch (const nlohmann::json::exception& e) {
		throw std::runtime_error("ERROR: sceneFile.cpp -> loadSceneFile(): " + path.string() + ": " + e.what());
	}

	if (scene.version != 1) {
		throw std::runtime_error("ERROR: sceneFile.cpp -> loadSceneFile(): unsupported version " + std::to_string(scene.version));
	}
	if (scene.image_size[0] == 0 || scene.image_size[1] == 0) {
		throw std::runtime_error("ERROR: sceneFile.cpp -> loadSceneFile(): image_size missing in " + path.string());
	}

	scene.directory = std::filesystem::absolute(path).parent_path();
	return scene;
}

void recast::saveSceneFile(const SceneFile& scene, const std::filesystem::path& path) {
	std::filesystem::path temporary = path;
	temporary += ".tmp";

	{
		std::ofstream file(temporary);
		if (!file.is_open()) {
			throw std::runtime_error("ERROR: sceneFile.cpp -> saveSceneFile(): can't write " + temporary.string());
		}
		file << nlohmann::json(scene).dump(2) << '\n';
	}

	std::filesystem::rename(temporary, path);
}

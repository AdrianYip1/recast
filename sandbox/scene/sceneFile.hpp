#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <engine/core/defines.hpp>

namespace recast {
	using Vec3f = std::array<foto::f32, 3>;

	// Pinhole camera in pixels. rotation is camera-to-world (rows), camera looks down -Z with +Y up
	struct SceneCamera {
		foto::f32 fx = 1000.0f;
		foto::f32 fy = 1000.0f;
		foto::f32 cx = 500.0f;
		foto::f32 cy = 500.0f;
		std::array<Vec3f, 3> rotation = { Vec3f{ 1, 0, 0 }, Vec3f{ 0, 1, 0 }, Vec3f{ 0, 0, 1 } };
		Vec3f position = { 0, 0, 0 };
	};

	struct ScaleReference {
		std::string kind = "camera_height";
		foto::f32 metres = 0.0f;
		foto::f32 guess = 0.0f; // the camera height first guessed (detect keeps its scale correction within 0.8-1.25x of it)
	};

	struct ObjectTransform {
		Vec3f position = { 0, 0, 0 };
		foto::f32 rotation_y_deg = 0.0f;
		foto::f32 scale = 1.0f;
	};

	struct SceneObject {
		std::string path;
		ObjectTransform transform;
	};

	struct ProxyMaterial {
		Vec3f albedo = { 0.5f, 0.5f, 0.5f };
		foto::f32 roughness = 0.9f;
	};

	// type "plane": point + normal. type "box": min + max (axis-aligned). type "hull": a convex shape fitted to a detected
	// object, vertices relative to point, faces counter-clockwise seen from outside, min + max its world bounding box
	struct Proxy {
		std::string type = "plane";
		foto::u32 id = 0;
		Vec3f point = { 0, 0, 0 };
		Vec3f normal = { 0, 1, 0 };
		Vec3f min = { 0, 0, 0 };
		Vec3f max = { 0, 0, 0 };
		ProxyMaterial material;
		std::vector<Vec3f> vertices;
		std::vector<std::array<foto::u32, 3>> faces;
	};

	// type "point": position + intensity (irradiance = intensity * cos / distance^2).
	// type "directional": direction (the way the light travels) + intensity (irradiance = intensity * cos).
	// type "area": corners + radiance (not rendered yet)
	struct SceneLight {
		std::string type = "point";
		Vec3f position = { 0, 0, 0 };
		Vec3f direction = { 0, -1, 0 };
		Vec3f intensity = { 0, 0, 0 };
		float radius = 0.05f;             // point: size of the bulb in metres, softens shadows
		float angular_radius_deg = 0.27f; // directional: the sun's apparent radius
		std::array<Vec3f, 4> corners = {};
		Vec3f radiance = { 0, 0, 0 };
	};

	struct Lighting {
		Vec3f ambient = { 0.2f, 0.2f, 0.2f };
		foto::u32 sh_order = 0;
		std::vector<Vec3f> sh;
	};

	struct SceneFile {
		int version = 1;
		std::string photo;
		std::array<foto::u32, 2> image_size = { 0, 0 };
		SceneCamera camera;
		ScaleReference scale_reference;
		SceneObject object;
		std::vector<Proxy> proxies;
		std::vector<SceneLight> lights;
		Lighting lighting;

		std::filesystem::path directory;

		std::string resolve(const std::string& relativePath) const;
	};

	SceneFile loadSceneFile(const std::filesystem::path& path);
	void saveSceneFile(const SceneFile& scene, const std::filesystem::path& path);
} // namespace recast

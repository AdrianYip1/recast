#include "gltfLoader.hpp"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <stb_image.h>

#include <filesystem>

using enginemath::Vec2;
using enginemath::Vec3;

namespace {
	struct CgltfDeleter {
		void operator()(cgltf_data* data) const { cgltf_free(data); }
	};

	struct Matrix3 {
		float m[3][3];

		Vec3 operator*(const Vec3& v) const {
			return Vec3(m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
						m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
						m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z);
		}

		float determinant() const {
			return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
				 - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
				 + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
		}

		Matrix3 inverseTranspose() const {
			float det = determinant();
			float inv = (det != 0.0f) ? 1.0f / det : 0.0f;
			Matrix3 r;
			r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv;
			r.m[0][1] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv;
			r.m[0][2] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv;
			r.m[1][0] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
			r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
			r.m[1][2] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
			r.m[2][0] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
			r.m[2][1] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
			r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
			return r;
		}
	};

	Vec3 transformPoint(const float world[16], const Vec3& p) {
		return Vec3(world[0] * p.x + world[4] * p.y + world[8] * p.z + world[12],
					world[1] * p.x + world[5] * p.y + world[9] * p.z + world[13],
					world[2] * p.x + world[6] * p.y + world[10] * p.z + world[14]);
	}

	Matrix3 upperLeft(const float world[16]) {
		Matrix3 r;
		for (int c = 0; c < 3; c++)
			for (int row = 0; row < 3; row++)
				r.m[c][row] = world[c * 4 + row];
		return r;
	}

	foto::ImageData decodeImage(const stbi_uc* bytes, size_t size) {
		int w = 0, h = 0, channels = 0;
		stbi_uc* pixels = stbi_load_from_memory(bytes, static_cast<int>(size), &w, &h, &channels, STBI_rgb_alpha);
		if (!pixels) return {};

		foto::ImageData image;
		image.width = static_cast<foto::u32>(w);
		image.height = static_cast<foto::u32>(h);
		image.rgba.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
		stbi_image_free(pixels);
		return image;
	}

	foto::ImageData loadImage(const cgltf_image& image, const cgltf_options& options, const std::filesystem::path& baseDir) {
		if (image.buffer_view) {
			return decodeImage(cgltf_buffer_view_data(image.buffer_view), image.buffer_view->size);
		}
		if (!image.uri) return {};

		std::string uri = image.uri;
		if (uri.rfind("data:", 0) == 0) {
			size_t comma = uri.find(',');
			if (comma == std::string::npos) return {};

			const char* base64 = uri.c_str() + comma + 1;
			size_t length = uri.size() - comma - 1;
			size_t padding = (length > 0 && base64[length - 1] == '=') + (length > 1 && base64[length - 2] == '=');
			size_t decodedSize = length / 4 * 3 - padding;

			void* decoded = nullptr;
			if (cgltf_load_buffer_base64(&options, decodedSize, base64, &decoded) != cgltf_result_success) return {};
			foto::ImageData result = decodeImage(static_cast<const stbi_uc*>(decoded), decodedSize);
			free(decoded);
			return result;
		}

		std::vector<char> decodedUri(uri.begin(), uri.end());
		decodedUri.push_back('\0');
		cgltf_decode_uri(decodedUri.data());

		std::ifstream file(baseDir / std::filesystem::u8path(decodedUri.data()), std::ios::binary);
		std::vector<stbi_uc> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		return bytes.empty() ? foto::ImageData{} : decodeImage(bytes.data(), bytes.size());
	}

	foto::Material convertMaterial(const cgltf_material& source, const cgltf_data& data) {
		foto::Material material;

		if (source.has_pbr_metallic_roughness) {
			const cgltf_pbr_metallic_roughness& pbr = source.pbr_metallic_roughness;
			material.baseColor = { pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2], pbr.base_color_factor[3] };
			material.metallic = pbr.metallic_factor;
			material.roughness = pbr.roughness_factor;

			const cgltf_texture* texture = pbr.base_color_texture.texture;
			material.baseColorTexture = (texture && texture->image)
				? static_cast<foto::u32>(texture->image - data.images)
				: foto::NO_TEXTURE;
		}
		else {
			material.baseColorTexture = foto::NO_TEXTURE;
		}

		float strength = source.has_emissive_strength ? source.emissive_strength.emissive_strength : 1.0f;
		material.emissive = { source.emissive_factor[0] * strength, source.emissive_factor[1] * strength,
							  source.emissive_factor[2] * strength, 0.0f };
		return material;
	}

	const cgltf_accessor* findAttribute(const cgltf_primitive& primitive, cgltf_attribute_type type) {
		for (cgltf_size i = 0; i < primitive.attributes_count; i++) {
			if (primitive.attributes[i].type == type && primitive.attributes[i].index == 0) {
				return primitive.attributes[i].data;
			}
		}
		return nullptr;
	}

	void appendPrimitive(foto::ModelData& model, const cgltf_primitive& primitive, const float world[16],
						 const cgltf_data& data, foto::u32& defaultMaterial) {
		if (primitive.type != cgltf_primitive_type_triangles) return;

		const cgltf_accessor* positions = findAttribute(primitive, cgltf_attribute_type_position);
		if (!positions || positions->count == 0) return;
		const cgltf_accessor* normals = findAttribute(primitive, cgltf_attribute_type_normal);
		const cgltf_accessor* uvs = findAttribute(primitive, cgltf_attribute_type_texcoord);

		foto::u32 baseVertex = static_cast<foto::u32>(model.vertices.size());
		foto::u32 firstIndex = static_cast<foto::u32>(model.indices.size());

		Matrix3 linear = upperLeft(world);
		Matrix3 normalMatrix = linear.inverseTranspose();
		bool mirrored = linear.determinant() < 0.0f;

		for (cgltf_size i = 0; i < positions->count; i++) {
			float p[3] = {}, n[3] = { 0.0f, 0.0f, 0.0f }, uv[2] = {};
			cgltf_accessor_read_float(positions, i, p, 3);
			if (normals) cgltf_accessor_read_float(normals, i, n, 3);
			if (uvs) cgltf_accessor_read_float(uvs, i, uv, 2);

			Vec3 normal = normalMatrix * Vec3(n[0], n[1], n[2]);
			model.vertices.push_back({
				.pos = transformPoint(world, Vec3(p[0], p[1], p[2])),
				.normal = (normal.dot(normal) > 0.0f) ? normal.normalized() : normal,
				.uv = Vec2(uv[0], uv[1])
			});
		}

		cgltf_size indexCount = primitive.indices ? primitive.indices->count : positions->count;
		for (cgltf_size i = 0; i + 2 < indexCount; i += 3) {
			foto::u32 tri[3];
			for (int k = 0; k < 3; k++) {
				cgltf_size index = primitive.indices ? cgltf_accessor_read_index(primitive.indices, i + k) : i + k;
				tri[k] = baseVertex + static_cast<foto::u32>(index);
			}
			if (mirrored) std::swap(tri[1], tri[2]);
			model.indices.insert(model.indices.end(), tri, tri + 3);
		}

		if (!normals) {
			for (size_t i = firstIndex; i < model.indices.size(); i += 3) {
				foto::Vertex& a = model.vertices[model.indices[i]];
				foto::Vertex& b = model.vertices[model.indices[i + 1]];
				foto::Vertex& c = model.vertices[model.indices[i + 2]];
				Vec3 face = (b.pos - a.pos).cross(c.pos - a.pos);
				a.normal += face;
				b.normal += face;
				c.normal += face;
			}
			for (size_t v = baseVertex; v < model.vertices.size(); v++) {
				Vec3& normal = model.vertices[v].normal;
				if (normal.dot(normal) > 0.0f) normal.normalize();
			}
		}

		foto::u32 materialIndex;
		if (primitive.material) {
			materialIndex = static_cast<foto::u32>(primitive.material - data.materials);
		}
		else {
			if (defaultMaterial == UINT32_MAX) {
				foto::Material fallback;
				fallback.baseColorTexture = foto::NO_TEXTURE;
				defaultMaterial = static_cast<foto::u32>(model.materials.size());
				model.materials.push_back(fallback);
			}
			materialIndex = defaultMaterial;
		}

		foto::u32 count = static_cast<foto::u32>(model.indices.size()) - firstIndex;
		if (count > 0) {
			model.submeshes.push_back({ .firstIndex = firstIndex, .indexCount = count, .materialIndex = materialIndex });
		}
	}

	void visitNode(foto::ModelData& model, const cgltf_node& node, const cgltf_data& data, foto::u32& defaultMaterial) {
		if (node.mesh) {
			float world[16];
			cgltf_node_transform_world(&node, world);
			for (cgltf_size p = 0; p < node.mesh->primitives_count; p++) {
				appendPrimitive(model, node.mesh->primitives[p], world, data, defaultMaterial);
			}
		}
		for (cgltf_size c = 0; c < node.children_count; c++) {
			visitNode(model, *node.children[c], data, defaultMaterial);
		}
	}
}

foto::ModelData foto::loadGltf(const std::string& filePath) {
	cgltf_options options = {};
	cgltf_data* raw = nullptr;

	if (cgltf_parse_file(&options, filePath.c_str(), &raw) != cgltf_result_success) {
		throw std::runtime_error("ERROR: gltfLoader.cpp -> loadGltf(): can't parse " + filePath);
	}
	std::unique_ptr<cgltf_data, CgltfDeleter> data(raw);

	if (cgltf_load_buffers(&options, data.get(), filePath.c_str()) != cgltf_result_success) {
		throw std::runtime_error("ERROR: gltfLoader.cpp -> loadGltf(): can't load buffers for " + filePath);
	}

	ModelData model;
	std::filesystem::path baseDir = std::filesystem::path(filePath).parent_path();

	for (cgltf_size i = 0; i < data->images_count; i++) {
		ImageData image = loadImage(data->images[i], options, baseDir);
		if (image.rgba.empty()) {
			std::cerr << "[gltf] couldn't decode image " << i << " in " << filePath << ", using white\n";
			image = { { 255, 255, 255, 255 }, 1, 1 };
		}
		model.images.push_back(std::move(image));
	}

	for (cgltf_size i = 0; i < data->materials_count; i++) {
		model.materials.push_back(convertMaterial(data->materials[i], *data));
	}

	u32 defaultMaterial = UINT32_MAX;
	const cgltf_scene* scene = data->scene ? data->scene : (data->scenes_count > 0 ? &data->scenes[0] : nullptr);
	if (scene) {
		for (cgltf_size n = 0; n < scene->nodes_count; n++) {
			visitNode(model, *scene->nodes[n], *data, defaultMaterial);
		}
	}
	else {
		for (cgltf_size n = 0; n < data->nodes_count; n++) {
			if (!data->nodes[n].parent) visitNode(model, data->nodes[n], *data, defaultMaterial);
		}
	}

	if (model.vertices.empty() || model.indices.empty()) {
		throw std::runtime_error("ERROR: gltfLoader.cpp -> loadGltf(): no triangle meshes in " + filePath);
	}

	return model;
}

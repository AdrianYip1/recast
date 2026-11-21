// Shared by pathtrace.comp and caustics.comp. Needs common.glsl (with CUSTOM_PUSH) and the ray query,
// buffer reference and nonuniform qualifier extensions first

layout(push_constant) uniform TracePush {
	uint sampleIndex;
	uint photoTexture;
	uint trace;
	uint photonCount;
	vec4 objectSphere; // bounding sphere of the dynamic casters (w = radius, 0 = none), where photons are aimed
	uint ownerTexture; // which cutout each photo pixel shows (Material.maskId), 0 = none
} push;

layout(set = 1, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 1, binding = 1, rgba32f) uniform image2D accumulation;
layout(set = 1, binding = 2, rgba16f) uniform writeonly image2D outputImage;

struct Geometry {
	uvec2 vertexAddress;
	uvec2 indexAddress;
	uint firstIndex;
	uint materialIndex;
	uint dynamic;
	uint pad;
};

layout(std430, set = 1, binding = 3) readonly buffer Geometries {
	Geometry geometries[];
};

// Denoising inputs (see denoise.comp): rgb = summed illumination (the part of each sample that isn't texture), a = summed
// squared luminance of it. Guide: the pixel's surface normal and distance, negative distance on objects, 0 = nothing hit
layout(set = 1, binding = 5, rgba32f) uniform image2D illumination;
layout(set = 1, binding = 6, rgba32f) uniform image2D guide;

// Caustic photons landing on each framebuffer pixel, summed since the last reset: 3 columns per pixel (rgb) in photonUnit()s
layout(set = 1, binding = 4, r32ui) uniform uimage2D caustics;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Floats { float f[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Indices { uint i[]; };

const uint MASK_STATIC = 0x01u;
const uint MASK_ALL = 0xFFu;
const uint VERTEX_FLOATS = 8u;
const float PI = 3.14159265;
const float SURFACE_OFFSET = 1e-3;

struct Hit {
	bool found;
	bool dynamic;
	bool frontFace; // the ray arrived from outside the surface (into a closed mesh)
	vec3 position;
	vec3 normal;
	vec2 uv;
	uint material;
};

uint hash(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

float random(inout uint state) {
	state = hash(state);
	return float(state >> 8) / 16777216.0;
}

vec3 cosineSample(vec3 normal, inout uint state) {
	float u = random(state);
	float phi = 2.0 * PI * random(state);
	vec3 tangent = normalize(cross(normal, abs(normal.x) > 0.5 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
	vec3 bitangent = cross(normal, tangent);
	float r = sqrt(u);
	return normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + normal * sqrt(max(0.0, 1.0 - u)));
}

Hit traceClosest(vec3 origin, vec3 direction, uint mask) {
	rayQueryEXT query;
	rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT, mask, origin, 1e-4, direction, 1e30);
	while (rayQueryProceedEXT(query)) {}

	Hit hit;
	hit.found = rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT;
	if (!hit.found) return hit;

	Geometry geometry = geometries[rayQueryGetIntersectionInstanceCustomIndexEXT(query, true) + rayQueryGetIntersectionGeometryIndexEXT(query, true)];
	Indices indices = Indices(geometry.indexAddress);
	Floats vertices = Floats(geometry.vertexAddress);
	uint first = geometry.firstIndex + 3u * uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true));

	vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(query, true);
	vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
	vec3 normal = vec3(0.0);
	vec2 uv = vec2(0.0);
	for (uint k = 0u; k < 3u; k++) {
		uint v = indices.i[first + k] * VERTEX_FLOATS;
		normal += weights[k] * vec3(vertices.f[v + 3u], vertices.f[v + 4u], vertices.f[v + 5u]);
		uv += weights[k] * vec2(vertices.f[v + 6u], vertices.f[v + 7u]);
	}

	normal = normalize(mat3(rayQueryGetIntersectionObjectToWorldEXT(query, true)) * normal);
	hit.frontFace = dot(normal, direction) < 0.0;
	if (!hit.frontFace) normal = -normal;

	hit.dynamic = geometry.dynamic != 0u;
	hit.position = origin + direction * rayQueryGetIntersectionTEXT(query, true);
	hit.normal = normal;
	hit.uv = uv;
	hit.material = geometry.materialIndex;
	return hit;
}

bool occluded(vec3 origin, vec3 direction, float maxDistance, uint mask) {
	rayQueryEXT query;
	rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, mask, origin, 1e-4, direction, maxDistance);
	while (rayQueryProceedEXT(query)) {}
	return rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

mat3 basis(vec3 normal) {
	vec3 tangent = normalize(cross(normal, abs(normal.x) > 0.5 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
	return mat3(tangent, cross(normal, tangent), normal);
}

vec3 sampleSphere(inout uint state) {
	float z = 1.0 - 2.0 * random(state);
	float phi = 2.0 * PI * random(state);
	float r = sqrt(max(0.0, 1.0 - z * z));
	return vec3(r * cos(phi), r * sin(phi), z);
}

vec3 sampleCone(vec3 axis, float angle, inout uint state) {
	float cosTheta = mix(1.0, cos(angle), random(state));
	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
	float phi = 2.0 * PI * random(state);
	return basis(axis) * vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

vec3 fresnel(vec3 f0, float cosine) {
	return f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0);
}

float smithG(float NdotL, float NdotV, float alpha) {
	float k = alpha * 0.5;
	return NdotL / (NdotL * (1.0 - k) + k) * NdotV / (NdotV * (1.0 - k) + k);
}

// Irradiance from the lights (in the photo's units: radiance = albedo * irradiance), each sampled at a random
// point of its shape so shadows soften. alpha > 0 also returns the glossy radiance reflected towards `view`.
// Calls with the same seed pick the same light points
void directLight(vec3 position, vec3 normal, vec3 view, uint mask, uint seed, float alpha, vec3 f0,
				 out vec3 irradiance, out vec3 specular) {
	irradiance = vec3(0.0);
	specular = vec3(0.0);
	uint count = frame.lightInfo.x;
	if (count == 0u) {
		irradiance = vec3(0.65 * abs(dot(normal, -frame.lightDir.xyz)));
		return;
	}

	vec3 origin = position + normal * SURFACE_OFFSET;
	for (uint i = 0u; i < count; i++) {
		Light light = frame.lights[i];
		vec3 direction;
		float distance;
		vec3 incoming;
		if (isDirectional(light)) {
			direction = sampleCone(-light.direction.xyz, light.direction.w, seed);
			distance = 1e30;
			incoming = light.intensity.rgb;
		}
		else if (isArea(light)) {
			// one random point on the rectangle: radiance * cos at the light * area / distance^2
			vec3 target = light.position.xyz + light.direction.xyz * (2.0 * random(seed) - 1.0)
						+ vec3(0.0, light.direction.w * (2.0 * random(seed) - 1.0), 0.0);
			vec3 toLight = target - position;
			distance = length(toLight);
			direction = toLight / max(distance, 1e-4);
			float cosLight = dot(areaNormal(light), -direction);
			if (cosLight <= 0.0) continue;
			incoming = light.intensity.rgb * (cosLight * areaSize(light) / max(distance * distance, 1e-4));
		}
		else {
			vec3 toLight = light.position.xyz + sampleSphere(seed) * light.direction.w - position;
			distance = length(toLight);
			direction = toLight / max(distance, 1e-4);
			incoming = light.intensity.rgb / max(distance * distance, 1e-4);
		}

		float NdotL = dot(normal, direction);
		if (NdotL <= 0.0) continue;
		if (light.position.w > 0.5 && occluded(origin, direction, distance, mask)) continue;
		irradiance += incoming * NdotL;

		if (alpha > 0.0) {
			float NdotV = max(dot(normal, view), 1e-4);
			vec3 halfway = normalize(direction + view);
			float NdotH = max(dot(normal, halfway), 0.0);
			float a2 = alpha * alpha;
			float denominator = NdotH * NdotH * (a2 - 1.0) + 1.0;
			float D = a2 / (PI * denominator * denominator);
			vec3 brdf = D * smithG(NdotL, NdotV, alpha) * fresnel(f0, max(dot(view, halfway), 0.0)) / (4.0 * NdotL * NdotV);
			specular += PI * brdf * incoming * NdotL;
		}
	}
}

vec3 directIrradiance(vec3 position, vec3 normal, uint mask, uint seed) {
	vec3 irradiance, specular;
	directLight(position, normal, vec3(0.0), mask, seed, 0.0, vec3(0.0), irradiance, specular);
	return irradiance;
}

vec3 albedoOf(Hit hit) {
	Material material = materials[hit.material];
	return material.baseColor.rgb * textureLod(textures[nonuniformEXT(material.baseColorTexture)], hit.uv, 0.0).rgb;
}

vec3 photoAt(vec2 viewportUV) {
	return textureLod(textures[push.photoTexture], viewportUV, 0.0).rgb;
}

// The cutout the photo shows at this pixel (exact, no filtering), 0 = none
uint ownerAt(vec2 viewportUV) {
	if (push.ownerTexture == 0u) return 0u;
	ivec2 size = textureSize(textures[push.ownerTexture], 0);
	ivec2 texel = clamp(ivec2(viewportUV * vec2(size)), ivec2(0), size - 1);
	return uint(texelFetch(textures[push.ownerTexture], texel, 0).r * 255.0 + 0.5);
}

// Diffuse light leaving an inserted object (no further bounces). Glass mostly lets the room's ambient light through
vec3 objectDiffuse(Hit hit, uint lightSeed) {
	Material material = materials[hit.material];
	vec3 diffuseColor = albedoOf(hit) * (1.0 - clamp(material.metallic, 0.0, 1.0));
	vec3 diffuse = diffuseColor * (frame.ambient.rgb + directIrradiance(hit.position, hit.normal, MASK_ALL, lightSeed));
	float transmission = clamp(material.transmission, 0.0, 1.0);
	return mix(diffuse, frame.ambient.rgb, transmission);
}

float dielectricFresnel(float cosine, float ior) {
	float f0 = (1.0 - ior) / (1.0 + ior);
	f0 *= f0;
	return f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0);
}

// Turns a ray at a glass surface: refracts (tinted by the glass colour when entering) or reflects by Fresnel.
// Returns the new origin; direction and throughput are updated in place
vec3 glassBounce(Hit hit, inout vec3 direction, inout vec3 throughput, inout uint state) {
	float ior = max(materials[hit.material].ior, 1.0);
	float eta = hit.frontFace ? 1.0 / ior : ior;
	float cosIn = clamp(dot(-direction, hit.normal), 0.0, 1.0);
	float sinOut2 = eta * eta * (1.0 - cosIn * cosIn);
	bool totalReflection = sinOut2 >= 1.0;
	float cosine = hit.frontFace ? cosIn : sqrt(max(0.0, 1.0 - sinOut2));
	if (totalReflection || random(state) < dielectricFresnel(cosine, ior)) {
		direction = reflect(direction, hit.normal);
		return hit.position + hit.normal * SURFACE_OFFSET;
	}
	direction = normalize(refract(direction, hit.normal, eta));
	if (hit.frontFace) throughput *= albedoOf(hit);
	return hit.position - hit.normal * SURFACE_OFFSET;
}

bool mirrorLike(Material material) {
	return material.metallic > 0.5 && material.roughness < 0.3;
}

// Photon flux unit: the largest flux a photon can start with, over 256. Photons from point lights are aimed at the
// objects' bounding sphere, photons from directional lights cross a disk that covers it
float photonUnit() {
	float largest = 0.0;
	uint count = frame.lightInfo.x;
	float perLight = float(push.photonCount) / float(max(count, 1u));
	for (uint i = 0u; i < count; i++) {
		Light light = frame.lights[i];
		float radius = push.objectSphere.w;
		float area;
		if (isDirectional(light)) {
			area = PI * radius * radius;
		}
		else {
			float distance = length(push.objectSphere.xyz - light.position.xyz);
			float sinMax = min(radius / max(distance, 1e-4), 1.0);
			area = 2.0 * PI * (1.0 - sqrt(1.0 - sinMax * sinMax));
		}
		vec3 strength = isDirectional(light) ? light.intensity.rgb : pointIntensity(light);
		largest = max(largest, max(strength.r, max(strength.g, strength.b)) * area / perLight);
	}
	return max(largest, 1e-12) / 256.0;
}


// Light leaving a photo surface: the photo itself where the camera sees that point, otherwise its albedo lit by the scene
vec3 roomRadiance(Hit hit, uint lightSeed) {
	vec4 clip = frame.proj * frame.view * vec4(hit.position, 1.0);
	if (clip.w > 0.0) {
		vec2 viewportUV = clip.xy / clip.w * 0.5 + 0.5;
		if (all(greaterThanEqual(viewportUV, vec2(0.0))) && all(lessThanEqual(viewportUV, vec2(1.0)))) {
			vec3 toPoint = hit.position - frame.cameraPos.xyz;
			float distance = length(toPoint);
			if (!occluded(frame.cameraPos.xyz, toPoint / distance, distance - 1e-2, MASK_STATIC)) return photoAt(viewportUV);
		}
	}
	return albedoOf(hit) * (frame.ambient.rgb + directIrradiance(hit.position, hit.normal, MASK_STATIC, lightSeed));
}


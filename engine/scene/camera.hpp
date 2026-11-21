#pragma once

#include <enginemath/enginemath.hpp>
#include <engine/core/defines.hpp>

namespace foto {
	// Perspective camera. World is right-handed and +Y up; yaw = 0, pitch = 0 looks down -Z.
	class Camera {
	public:
		enginemath::Vec3 position = { 0.0f, 0.0f, 3.0f };
		f32 yaw = 0.0f;                              // radians, positive turns right (towards +X)
		f32 pitch = 0.0f;                            // radians, positive looks up, clamped to about +-89 degrees
		f32 fovY = enginemath::toRad(60.0f);         // vertical field of view, radians
		f32 nearPlane = 0.05f;
		f32 farPlane = 1000.0f;

		enginemath::Vec3 getForward() const;
		enginemath::Vec3 getRight() const;
		enginemath::Vec3 getUp() const;

		enginemath::Mat4 getView() const;
		// Vulkan clip space: Y flipped (points down) and depth 0..1
		enginemath::Mat4 getProjection(f32 aspect) const;

		// Points the camera at a world position (sets yaw + pitch)
		void lookAt(const enginemath::Vec3& target);

		// Keeps pitch away from straight up/down, where the view matrix breaks down
		void clampPitch();
	};
} // namespace foto

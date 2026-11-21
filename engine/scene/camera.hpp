#pragma once

#include <optional>

#include <enginemath/enginemath.hpp>
#include <engine/core/defines.hpp>

namespace foto {
	struct CameraIntrinsics {
		f32 fx = 0.0f;
		f32 fy = 0.0f;
		f32 cx = 0.0f;
		f32 cy = 0.0f;
		f32 width = 0.0f;
		f32 height = 0.0f;
	};

	// Perspective camera. World is right-handed and +Y up; yaw = 0, pitch = 0 looks down -Z.
	// With intrinsics + orientation set it reproduces a real photo's pinhole camera exactly.
	class Camera {
	public:
		enginemath::Vec3 position = { 0.0f, 0.0f, 3.0f };
		f32 yaw = 0.0f;                              // radians, positive turns right (towards +X)
		f32 pitch = 0.0f;                            // radians, positive looks up, clamped to about +-89 degrees
		f32 fovY = enginemath::toRad(60.0f);         // vertical field of view, radians (ignored with intrinsics)
		f32 nearPlane = 0.05f;
		f32 farPlane = 1000.0f;

		enginemath::Vec3 getForward() const;
		enginemath::Vec3 getRight() const;
		enginemath::Vec3 getUp() const;

		enginemath::Mat4 getView() const;
		// Vulkan clip space: Y flipped (points down) and depth 0..1
		enginemath::Mat4 getProjection(f32 aspect) const;

		// World-space direction through a point of the view, in NDC (-1..1, y down like Vulkan)
		enginemath::Vec3 rayDirection(f32 ndcX, f32 ndcY, f32 aspect) const;

		void lookAt(const enginemath::Vec3& target);
		void clampPitch();

		// Pixel-space pinhole intrinsics; the viewport should match width/height's aspect
		void setIntrinsics(const CameraIntrinsics& intrinsics);
		void clearIntrinsics();
		const std::optional<CameraIntrinsics>& getIntrinsics() const { return intrinsics; };

		// Camera-to-world rotation (columns = camera right, up, back). Overrides yaw/pitch until cleared
		void setOrientation(const enginemath::Mat4& cameraToWorld);
		// Drops the fixed orientation, keeping the current view direction as yaw/pitch (roll is lost)
		void clearOrientation();
		bool hasOrientation() const { return orientation.has_value(); };

	private:
		std::optional<CameraIntrinsics> intrinsics;
		std::optional<enginemath::Mat4> orientation;
	};
} // namespace foto

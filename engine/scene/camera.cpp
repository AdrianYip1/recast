#include "camera.hpp"

#include <algorithm>
#include <cmath>

using enginemath::Vec3;
using enginemath::Mat4;

Vec3 foto::Camera::getForward() const {
	return Vec3(std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch));
}

Vec3 foto::Camera::getRight() const {
	return getForward().cross(Vec3::up()).normalized();
}

Vec3 foto::Camera::getUp() const {
	return getRight().cross(getForward()).normalized();
}

Mat4 foto::Camera::getView() const {
	return Mat4::lookAtM(position, position + getForward(), Vec3::up());
}

Mat4 foto::Camera::getProjection(f32 aspect) const {
	// enginemath builds a GL-style (y up) projection; ENGINEMATH_DEPTH_ZERO_TO_ONE already gives 0..1 depth
	Mat4 projection = Mat4::projectionM(fovY, aspect, nearPlane, farPlane);
	projection.m[1][1] *= -1.0f;
	return projection;
}

void foto::Camera::lookAt(const Vec3& target) {
	Vec3 direction = (target - position).normalized();
	pitch = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
	yaw = std::atan2(direction.x, -direction.z);
	clampPitch();
}

void foto::Camera::clampPitch() {
	constexpr f32 limit = enginemath::toRad(89.0f);
	pitch = std::clamp(pitch, -limit, limit);
}

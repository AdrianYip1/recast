#include "camera.hpp"

#include <algorithm>
#include <cmath>

using enginemath::Vec3;
using enginemath::Vec4;
using enginemath::Mat4;

Vec3 foto::Camera::getForward() const {
	if (orientation) {
		return Vec3(-orientation->m[2][0], -orientation->m[2][1], -orientation->m[2][2]);
	}
	return Vec3(std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch));
}

Vec3 foto::Camera::getRight() const {
	if (orientation) {
		return Vec3(orientation->m[0][0], orientation->m[0][1], orientation->m[0][2]);
	}
	return getForward().cross(Vec3::up()).normalized();
}

Vec3 foto::Camera::getUp() const {
	if (orientation) {
		return Vec3(orientation->m[1][0], orientation->m[1][1], orientation->m[1][2]);
	}
	return getRight().cross(getForward()).normalized();
}

Mat4 foto::Camera::getView() const {
	if (!orientation) {
		return Mat4::lookAtM(position, position + getForward(), Vec3::up());
	}

	Mat4 view = Mat4::identity();
	for (int c = 0; c < 3; c++) {
		for (int r = 0; r < 3; r++) {
			view.m[c][r] = orientation->m[r][c];
		}
	}
	for (int r = 0; r < 3; r++) {
		view.m[3][r] = -(orientation->m[r][0] * position.x + orientation->m[r][1] * position.y + orientation->m[r][2] * position.z);
	}
	return view;
}

Mat4 foto::Camera::getProjection(f32 aspect) const {
	if (!intrinsics) {
		Mat4 projection = Mat4::projectionM(fovY, aspect, nearPlane, farPlane);
		projection.m[1][1] *= -1.0f;
		return projection;
	}

	const CameraIntrinsics& k = *intrinsics;
	Mat4 projection = Mat4::zero();
	projection.m[0][0] = 2.0f * k.fx / k.width;
	projection.m[1][1] = -2.0f * k.fy / k.height;
	projection.m[2][0] = 1.0f - 2.0f * k.cx / k.width;
	projection.m[2][1] = 1.0f - 2.0f * k.cy / k.height;
	projection.m[2][2] = -farPlane / (farPlane - nearPlane);
	projection.m[2][3] = -1.0f;
	projection.m[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
	return projection;
}

void foto::Camera::lookAt(const Vec3& target) {
	orientation.reset();
	Vec3 direction = (target - position).normalized();
	pitch = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
	yaw = std::atan2(direction.x, -direction.z);
	clampPitch();
}

void foto::Camera::clampPitch() {
	constexpr f32 limit = enginemath::toRad(89.0f);
	pitch = std::clamp(pitch, -limit, limit);
}

void foto::Camera::setIntrinsics(const CameraIntrinsics& value) {
	intrinsics = value;
}

void foto::Camera::clearIntrinsics() {
	intrinsics.reset();
}

void foto::Camera::setOrientation(const Mat4& cameraToWorld) {
	orientation = cameraToWorld;
}

void foto::Camera::clearOrientation() {
	if (!orientation) return;
	Vec3 forward = getForward();
	orientation.reset();
	pitch = std::asin(std::clamp(forward.y, -1.0f, 1.0f));
	yaw = std::atan2(forward.x, -forward.z);
	clampPitch();
}

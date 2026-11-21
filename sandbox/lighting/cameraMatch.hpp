#pragma once

#include "lighting/lightFit.hpp"

namespace recast {
	// How the photo's camera degrades an image, so a rendered object can be degraded the same way
	struct CameraMatch {
		float grain = 0.0f; // noise standard deviation in sqrt(linear) units (shot noise grows with brightness)
		float blur = 0.0f;  // gaussian sigma of the edge softness, in photo pixels
	};

	// Grain: Immerkaer's noise estimate on flat areas. Blur: spread of the brightness step across the photo's sharpest edges
	CameraMatch estimateCameraMatch(const LinearImage& photo);
} // namespace recast

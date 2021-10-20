#pragma once

#include <UGM/UGM.h>

#include "../Eigen/Dense."

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool adding_line{ false };

	bool enable01{ true };
	bool enable02{ true };
	bool enable03{ true };
	bool enable04{ true };

	int highest = 1;

	std::vector<float> coordx;
	std::vector<float> coordy1;
	std::vector<float> coordy2;
	std::vector<float> coordy3;
	std::vector<float> coordy4;
	float lamda = 1.0f;
};

#include "details/CanvasData_AutoRefl.inl"

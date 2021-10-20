#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include "spdlog/spdlog.h"

using namespace Ubpa;

std::vector<float> lagrange(std::vector<Ubpa::pointf2> input_points, std::vector<float> xs )
{
	int length = input_points.size();
	std::vector<float> ys;
	if (length >= 2)
	{
		for (auto x : xs)
		{
			float y = 0.0f;
			for (int i = 0; i < length; i++)
			{
				float temp = input_points[i][1];
				for (int j = 0; j < length; j++)
				{
					if (i != j)
					{
						temp *= x - input_points[j][0];
						temp *= 1 / (input_points[i][0] - input_points[j][0]);
					}
				}
				y += temp;
			}
			ys.push_back(y);
		}
		
	}
	return ys;
}

std::vector<float> Gauss(std::vector<Ubpa::pointf2> input_points, std::vector<float> xs)
{
	int length = input_points.size();
	std::vector<float> ys;
	Eigen::MatrixXf A(length, length);
	Eigen::VectorXf b(length);
	Eigen::VectorXf x(length);
	float inv_sigma_sq = 1.0 / pow(100.0, 2);
	if (length >= 2)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < length; j++)
				if (i == j) A(i, j) = 1;
				else A(i, j) = exp(-0.5 * pow(input_points[j][0] - input_points[i][0], 2) * inv_sigma_sq);
			b(i) = input_points[i][1];
		}
		x = A.colPivHouseholderQr().solve(b);

		for (auto xcoord : xs)
		{
			float y = 0.0f;
			for (int i = 0; i < length; i++)
				y += x(i) * exp(-0.5 * pow(xcoord - input_points[i][0], 2) * inv_sigma_sq);
			ys.push_back(y);
		}
	}
	return ys;
}

std::vector<float> LSM(std::vector<Ubpa::pointf2> input_points, int highest, std::vector<float> xs) {
	int length = input_points.size();
	highest = length - 2 > highest ? highest : length - 2;
	std::vector<float> ys;
	Eigen::MatrixXf A(length, highest + 1);
	Eigen::MatrixXf ATA(highest + 1, highest + 1);
	Eigen::VectorXf b(length);
	Eigen::VectorXf ATb(highest + 1);
	Eigen::VectorXf x(highest + 1);
	if (length > 2)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < highest + 1; j++)
				A(i, j) = pow(input_points[i][0], j);
			b(i) = input_points[i][1];
		}
		ATA = A.transpose() * A;
		ATb = A.transpose() * b;
		x = ATA.inverse() * ATb;

		for (auto xcoord : xs)
		{
			float y = x(0);
			for (int i = 1; i < highest + 1; i++)
				y += x(i) * pow(xcoord, i);
			ys.push_back(y);
		}
	}
	return ys;
}

std::vector<float> RR(std::vector<Ubpa::pointf2> input_points, int highest, float lamda, std::vector<float> xs) {
	int length = input_points.size();
	highest = length - 2 > highest ? highest : length - 2;
	std::vector<float> ys;
	Eigen::MatrixXf A(length, highest + 1);
	Eigen::MatrixXf ATA_add(highest + 1, highest + 1);

	Eigen::VectorXf b(length);
	Eigen::VectorXf ATb(highest + 1);
	Eigen::VectorXf x(highest + 1);
	if (length > 2)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < highest + 1; j++)
				A(i, j) = pow(input_points[i][0], j);
			b(i) = input_points[i][1];
		}
		ATA_add = A.transpose() * A + lamda * Eigen::MatrixXf::Identity(highest + 1, highest + 1);
		ATb = A.transpose() * b;
		x = ATA_add.inverse() * ATb;

		for (auto xcoord : xs)
		{
			float y = x(0);
			for (int i = 1; i < highest + 1; i++)
				y += x(i) * pow(xcoord, i);
			ys.push_back(y);
		}
	}
	return ys;
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: click to add points,\nMouse Right: drag to scroll, click for context menu.");
			
			ImGui::Checkbox("lagrange", &data->enable01);
			ImGui::SameLine(120);
			ImGui::Checkbox("Gauss", &data->enable02);
			ImGui::Checkbox("Least Squares Method", &data->enable03);
			ImGui::SameLine(250);
			ImGui::SliderInt("Highest Power", &data->highest, 1, 10);
			ImGui::Checkbox("Ridge Regression", &data->enable04);
			ImGui::SameLine(250);
			ImGui::SliderFloat("Lamda", &data->lamda, 0.0f, 100.0f);

			// Typically you would use a BeginChild()/EndChild() pair to benefit from a clipping region + own scrolling.
			// Here we demonstrate that this can be replaced by simple offsetting + custom drawing + PushClipRect/PopClipRect() calls.
			// To use a child window instead we could use, e.g:
			//      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));      // Disable padding
			//      ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(50, 50, 50, 255));  // Set a background color
			//      ImGui::BeginChild("canvas", ImVec2(0.0f, 0.0f), true, ImGuiWindowFlags_NoMove);
			//      ImGui::PopStyleColor();
			//      ImGui::PopStyleVar();
			//      [...]
			//      ImGui::EndChild();

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
			const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);

			// Add first and second point
			//if (is_hovered && !data->adding_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			//{
			//	data->points.push_back(mouse_pos_in_canvas);
			//	data->points.push_back(mouse_pos_in_canvas);
			//	data->adding_line = true;
			//}
			//if (data->adding_line)
			//{
			//	data->points.back() = mouse_pos_in_canvas;
			//	if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
			//		data->adding_line = false;
			//}

			if (is_active && !data->adding_line && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				data->points.push_back(mouse_pos_in_canvas);
				spdlog::info("Point added at: {}, {}", data->points.back()[0], data->points.back()[1]);
				data->adding_line = true;
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
			{
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context"))
			{
				if (data->adding_line)
					data->points.resize(data->points.size() - 2);
				data->adding_line = false;
				if (ImGui::MenuItem("Remove one", NULL, false, data->points.size() > 0)) 
				{ 
					data->points.resize(data->points.size() - 1); 
					data->adding_line = true;
				}
				if (ImGui::MenuItem("Remove all", NULL, false, data->points.size() > 0)) 
				{ 
					data->points.clear(); 
					data->adding_line = true;
				}
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid)
			{
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));	
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
				draw_list->AddLine(ImVec2(origin.x, canvas_p0.y), ImVec2(origin.x, canvas_p1.y), IM_COL32(200, 200, 200, 160));
				draw_list->AddLine(ImVec2(canvas_p0.x, origin.y), ImVec2(canvas_p1.x, origin.y), IM_COL32(200, 200, 200, 160));
			}
			for (int n = 0; n < data->points.size(); n += 1)
				draw_list->AddCircleFilled(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), 3.0f, IM_COL32(255, 255, 0, 255), 6);
			draw_list->PopClipRect();

			
			if (data->adding_line)
			{
				data->coordx.clear();
				const float step_size = 1.0f;
				const float margin = 2.0f;
				for (float px = canvas_p0.x - origin.x + margin; px < canvas_p1.x - origin.x - margin; px += step_size)
				{
					data->coordx.push_back(px);
				}
				if (data->enable01)
					data->coordy1 = lagrange(data->points, data->coordx);
				if (data->enable02)
					data->coordy2 = Gauss(data->points, data->coordx);
				if (data->enable03 && data->points.size() > 2)
					data->coordy3 = LSM(data->points, data->highest, data->coordx);
				if (data->enable04 && data->points.size() > 2)
					data->coordy4 = RR(data->points, data->highest, data->lamda, data->coordx);
				data->adding_line = false;
			}

			if (data->points.size() > 1 && data->enable01)
			{
				int count = data->coordx.size();
				for (int i = 0; i < count - 1; i++)
				{
					ImVec2 start = ImVec2(data->coordx.at(i) + origin.x, data->coordy1.at(i) + origin.y);
					ImVec2 end = ImVec2(data->coordx.at(i + 1) + origin.x, data->coordy1.at(i + 1) + origin.y);
					//if (end.y - canvas_p1.y < margin && end.y - canvas_p0.y > margin &&
					//	start.y - canvas_p1.y < margin && start.y - canvas_p0.y > margin)
					draw_list->AddLine(start, end, IM_COL32(200, 10, 10, 255));
				}
			}

			if (data->points.size() > 1 && data->enable02)
			{
				int count = data->coordx.size();
				for (int i = 0; i < count - 1; i++)
				{
					ImVec2 start = ImVec2(data->coordx.at(i) + origin.x, data->coordy2.at(i) + origin.y);
					ImVec2 end = ImVec2(data->coordx.at(i + 1) + origin.x, data->coordy2.at(i + 1) + origin.y);
					draw_list->AddLine(start, end, IM_COL32(200, 10, 200, 255));
				}
			}

			if (data->points.size() > 2 && data->enable03)
			{
				int count = data->coordx.size();
				for (int i = 0; i < count - 1; i++)
				{
					ImVec2 start = ImVec2(data->coordx.at(i) + origin.x, data->coordy3.at(i) + origin.y);
					ImVec2 end = ImVec2(data->coordx.at(i + 1) + origin.x, data->coordy3.at(i + 1) + origin.y);
					draw_list->AddLine(start, end, IM_COL32(50, 200, 50, 255));
				}
			}

			if (data->points.size() > 2 && data->enable04)
			{
				int count = data->coordx.size();
				for (int i = 0; i < count - 1; i++)
				{
					ImVec2 start = ImVec2(data->coordx.at(i) + origin.x, data->coordy4.at(i) + origin.y);
					ImVec2 end = ImVec2(data->coordx.at(i + 1) + origin.x, data->coordy4.at(i + 1) + origin.y);
					draw_list->AddLine(start, end, IM_COL32(50, 50, 200, 255));
				}
			}
		}

		ImGui::End();
	});
}

#include <bits/stdint-uintn.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "adam.h"
#include "cifar10.h"

constexpr const uint32_t num_samples{1185};
constexpr const uint32_t image_height{256};
constexpr const uint32_t image_width{256};
constexpr const uint32_t image_channels{3};

std::vector<std::string> split_string(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

torch::Tensor read_single_adam_image(const std::string root, const std::string image_name)
{
	std::string image_type;
	if (image_name.at(0) == 'A')
		image_type = "AMD";
	else
		image_type = "Non-AMD";
	std::string image_path = join_paths(join_paths(root, image_type), image_name);

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::resize(img, img, {image_height, image_width});
    torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    return tensor.permute({2, 0, 1});
}

std::pair<torch::Tensor, torch::Tensor> read_data(std::string root)
{
	std::string filepath = join_paths(root, "fovea_location.csv");
	std::ifstream filehand(filepath);
	std::string line;

	torch::Tensor images = torch::zeros(
		{num_samples, image_channels, image_height, image_width},
		torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor bboxes = torch::zeros({num_samples, 2},
		torch::TensorOptions().dtype(torch::kFloat32));

	if (!filehand.is_open())
	{
        std::cerr << "Failed to open " << filepath << std::endl;
        // TORCH_CHECK(f, "Error opening images file at ", file_path);
    }

	bool first_line = true;
	int sample_index = 0;
	while (filehand)
	{
		std::getline(filehand, line);
		if (first_line || line.length() == 0)
		{
			first_line = false;
			continue; // omitting column names and last empty line
		}
		std::vector<std::string> tokens = split_string(line, ',');

		torch::Tensor cast = read_single_adam_image(root, tokens.at(1)).to(torch::kFloat32);
		images[sample_index] = cast.div_(255);
		bboxes[sample_index++] = torch::tensor({stod(tokens.at(2)), stod(tokens.at(3))}).div_((int)image_height);
	}
	std::cout << sample_index << std::endl;
	assert(
        (sample_index == num_samples) &&
        "Insufficient number of images. Data file might have been corrupted.");

	return std::make_pair(images, bboxes);
}
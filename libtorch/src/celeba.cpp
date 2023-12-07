#include <bits/stdint-uintn.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "celeba.h"
#include "cifar10.h"

const std::string inner_root{"img_align_celeba"};

// constexpr const uint32_t num_samples{202599};
constexpr const uint32_t num_samples{12600}; // temporary num since reading in takes too long
constexpr const uint32_t image_height{64};
constexpr const uint32_t image_width{64};
constexpr const uint32_t image_channels{3};

constexpr const uint32_t num_progress_bars{10};
constexpr const uint32_t bar_width{num_samples / num_progress_bars};

void draw_progress_bar(int count)
{
    int bars = count / bar_width;
    std::cout << "Reading images: [";
    for (int i = 0; i < num_progress_bars; i++)
    {
        if (i < bars)
            std::cout << "=";
        else
            std::cout << " ";
    }
    std::cout << "] " << count << " / " << num_samples << "\r";
    std::cout.flush();
}

torch::Tensor read_single_image(const std::string image_path)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::resize(img, img, {image_height, image_width});
    torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, at::kByte);
    return tensor.permute({2, 0, 1});
}

torch::Tensor read_images_from_root(std::string root)
{
    root = join_paths(root, inner_root);
    if (!std::filesystem::exists(root))
    {
        throw std::invalid_argument(root + " is not a valid root directory.");
    }

    std::vector<torch::Tensor> img_tensors;
    int count = 0;
    for (const auto &img_hand : std::filesystem::directory_iterator(root))
    {
        img_tensors.push_back(read_single_image(img_hand.path()));

        // drawing progress bar since it takes a while
        draw_progress_bar(++count);
    }
    std::cout << std::endl;

    torch::Tensor images = torch::stack(img_tensors);
    assert(
        (images.sizes()[0] == num_samples) &&
        "Insufficient number of images. Data file might have been corrupted.");

    return images;
}

torch::Tensor read_images(const std::string &root)
{
    torch::Tensor images = read_images_from_root(root);
    images = images.to(torch::kFloat32).div_(255);
    return images;
}

CELEBA::CELEBA(const std::string &root)
    : images_(read_images(root)),
      targets_(torch::zeros({num_samples}))
{
}

torch::data::Example<> CELEBA::get(size_t index)
{
    return {images_[index], targets_[index]};
}

torch::optional<size_t> CELEBA::size() const { return images_.size(0); }

bool CELEBA::is_train() const noexcept { return true; }

const torch::Tensor &CELEBA::images() const { return images_; }

const torch::Tensor &CELEBA::targets() const { return targets_; }
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

// constexpr const uint32_t num_samples{202599}; // full celeba dataset
// constexpr const uint32_t num_samples{12600}; // temporary since reading in takes too long
constexpr const uint32_t num_samples{15000}; // truncated celeba since full ds does not fit into my ram
constexpr const uint32_t image_height{64};
constexpr const uint32_t image_width{64};
constexpr const uint32_t image_channels{3};

constexpr const uint32_t num_progress_bars{20};
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

    torch::Tensor result = torch::zeros({num_samples, 3, 64, 64});
    int sample_index = 0;
    for (const auto &img_hand : std::filesystem::directory_iterator(root))
    {
        result[sample_index++] = read_single_image(img_hand.path());

        // drawing progress bar since it takes a while
        draw_progress_bar(sample_index);
    }
    std::cout << std::endl;

    assert(
        (sample_index == num_samples) &&
        "Insufficient number of images. Data file might have been corrupted.");

    return result;
}

torch::Tensor read_images(const std::string &root)
{
    torch::Tensor images = read_images_from_root(root);
    images = images.to(torch::kFloat32).div_(255);
    return images;
}

CELEBA::CELEBA(const std::string &root, const int batch_size)
    : images_(read_images(root)),
      targets_(torch::zeros({num_samples})),
      batch_size_(batch_size)
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

int CELEBA::get_max_batch_id() { return num_samples / batch_size_ - 1; }

torch::Tensor CELEBA::get_batch_by_id(int batch_id)
{
    // ensure validity of batch_id
    int max_batch_id = num_samples / batch_size_ - 1;
    max_batch_id = num_samples % batch_size_ == 0 ? max_batch_id : max_batch_id + 1;
    batch_id = batch_id > max_batch_id ? max_batch_id : batch_id;

    int batch_size_used = batch_size_;
    // ensure validity batch size if last batch
    if (batch_id == max_batch_id)
        batch_size_used = num_samples - batch_size_ * batch_id;

    torch::Tensor batch = images_.narrow(0, batch_id * batch_size_, batch_size_used);
    return batch;
}
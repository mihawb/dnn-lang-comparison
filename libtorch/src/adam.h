#ifndef _ADAM_H_
#define _ADAM_H_

#include <string>
#include <torch/torch.h>

std::pair<torch::Tensor, torch::Tensor> read_data(std::string root);

class ADAM : public torch::data::datasets::Dataset<ADAM>
{
public:
    enum class Mode
    {
        kTrain,
        kTest
    };

    explicit ADAM(const std::string &root, Mode mode = Mode::kTrain, float test_size = 0.8);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    bool is_train() const noexcept;

    const torch::Tensor &images() const;

    const torch::Tensor &targets() const;

private:
     torch::Tensor images_, targets_;
     int cutoff_;
};

#endif
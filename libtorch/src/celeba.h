#ifndef _CELEBA_H_
#define _CELEBA_H_

#include <string>
#include <torch/torch.h>

class CELEBA : public torch::data::datasets::Dataset<CELEBA>
{
public:
    explicit CELEBA(const std::string &root);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    bool is_train() const noexcept;

    const torch::Tensor &images() const;

    const torch::Tensor &targets() const;

private:
    torch::Tensor images_, targets_;
};

#endif
#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  
  if (torch::cuda::is_available()) {
	std::cout << "CUDA is available!" << std::endl;
  }
}

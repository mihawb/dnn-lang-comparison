// inicjalizacja modelu
std::shared_ptr<torch::nn::Module> model1 = std::make_shared<torch::nn::Module>();

torch::jit::script::Module model2_to_be_moved = torch::jit::load("model.pt");
std::shared_ptr<torch::jit::script::Module> model2 =
	std::make_shared<torch::jit::script::Module>(model2_to_be_moved);

// propagacja w przód
torch::Tensor input = ...;
torch::Tensor output1 = model1->forward(input);

torch::jit::Stack<torch::jit::IValue> input_ivalues;
input_ivalues.push_back(input);
torch::jit::IValue output2_ivalues = model2->forward(input_ivalues);
torch::Tensor output2 = output2_ivalues.toTensor();
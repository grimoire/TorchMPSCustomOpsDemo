#include <torch/extension.h>

using at::Tensor;
Tensor custom_add(const Tensor &a, const Tensor &b);
Tensor custom_add2(const Tensor &a, const Tensor &b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("custom_add", &custom_add, "custom add for mps test", py::arg("a"),py::arg("b"));
    m.def("custom_add2", &custom_add2, "custom add for mps test", py::arg("a"),py::arg("b"));
}
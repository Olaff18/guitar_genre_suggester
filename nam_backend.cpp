#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

class NAMWrapper {
public:
    NAMWrapper() {}

    void load(const std::string &nam_path, const std::string &ir_path) {
        // Load NAM + IR here
    }

    py::array_t<float> process(py::array_t<float> input) {
        auto buf = input.request();
        float *ptr = static_cast<float*>(buf.ptr);

        std::vector<float> out;
        out.resize(buf.size);
        std::copy(ptr, ptr + buf.size, out.begin());

        // Do processing (stub)
        for (auto &v : out) v *= 0.5f;

        // Return as NumPy array (copy)
        return py::array_t<float>(out.size(), out.data());
    }
};

PYBIND11_MODULE(nam_backend, m) {
    py::class_<NAMWrapper>(m, "NAMWrapper")
        .def(py::init<>())
        .def("load", &NAMWrapper::load)
        .def("process", &NAMWrapper::process);
}

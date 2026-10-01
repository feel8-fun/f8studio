from conan import ConanFile


class F8AudiocapDependencies(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("nlohmann_json/3.12.0")
        self.requires("cxxopts/3.3.1")
        self.requires("spdlog/1.16.0")

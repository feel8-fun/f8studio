from conan import ConanFile


class F8CppengineDependencies(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("nlohmann_json/3.12.0")
        self.requires("cxxopts/3.3.1")
        self.requires("spdlog/1.16.0")
        self.requires("mexce/1.0.1")
        self.requires("pybind11/3.0.1")
        self.requires("luajit/2.1.0-beta3")
        self.requires("sol2/3.5.0")

    def configure(self):
        self.options["sol2"].with_lua = "luajit"

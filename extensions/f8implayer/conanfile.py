from conan import ConanFile


class F8ImplayerDependencies(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("nlohmann_json/3.12.0")
        self.requires("cxxopts/3.3.1")
        self.requires("spdlog/1.16.0")
        self.requires("iconfontcppheaders/cci.20240620")
        self.requires("glad/0.1.36")
        self.requires("opengl/system")
        self.requires("imgui/1.92.4")
        self.requires("libpng/1.6.50")

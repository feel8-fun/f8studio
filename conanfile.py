from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.files import copy, rmdir
import os


class F8Build(ConanFile):
    # Basic metadata (assumed; adjust if you have canonical values)
    name = "f8Build"
    version = "0.1.0"
    license = "MIT"
    url = "https://github.com/feel8-fun/f8build"
    description = "Flow-based task engine"
    topics = ("ipc", "shared-memory", "synchronization")
    package_type = "library"

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    # Export the project sources so Conan can build from them. We include examples/tests
    # because they are small and sometimes users want to build them; the recipe only
    # pulls optional deps when requested.
    exports_sources = "src/*", "CMakeLists.txt", "examples/*", "tests/*"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_tests": [True, False],
        "with_examples": [True, False],
        "with_apps": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_tests": False,
        "with_examples": False,
        "with_apps": False
    }

    def config_options(self):
        # fPIC is irrelevant on Windows
        if self.settings.os == "Windows":
            try:
                del self.options.fPIC
            except Exception:
                pass

        self.options["sol2"].with_lua = "luajit"
        # Ensure OpenCV contrib modules are available for CV services.
        # (e.g. opencv_tracking for CSRT/KCF trackers)
        self.options["opencv"].tracking = True

    def configure(self):
        # When building shared libs, fPIC option is not needed
        if self.options.get_safe("shared"):
            try:
                del self.options.fPIC
            except Exception:
                pass

    def build_requirements(self):
        # Ensure a modern CMake is available as a build tool when Conan runs the build
        # Adjust the version to whatever your CI or environment prefers.
        # Using tool_requires (Conan 2) to declare build tools
        self.tool_requires("cmake/3.27.5")

    def requirements(self):
        # core runtime dependencies
        self.requires("cnats/3.11.0")
        self.requires("nlohmann_json/3.12.0")
        self.requires("openssl/3.6.0")
        self.requires("cxxopts/3.3.1")
        self.requires("spdlog/1.16.0")
        # Mathematical expressions plugin
        self.requires("mexce/1.0.1")
        # Python plugin
        self.requires("pybind11/3.0.1")
        # Lua plugin
        self.requires("luajit/2.1.0-beta3")
        self.requires("sol2/3.5.0")
        # AngelScript plugin
        self.requires("angelscript/2.38.0")
        # JS intepreter
        self.requires("jerryscript/2.4.0")
        self.requires("glm/1.0.1")

        # Video Player
        self.requires("libmpv/20251124")
        self.requires("ytdlp/2025.11.12")
        self.requires("iconfontcppheaders/cci.20240620")
        self.requires("opencv/4.12.0")
        self.requires("sdl/3.2.20")
        self.requires("glad/0.1.36")
        self.requires("opengl/system")
        self.requires("imgui/1.92.4")
        self.requires("pulseaudio/17.0", override=True)

        if self.options.with_tests:
            self.requires("gtest/1.17.0")

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        options = {
            "BUILD_SHARED_LIBS": "ON" if self.options.shared else "OFF",
            "BUILD_TESTS": "ON" if self.options.with_tests else "OFF",
            "BUILD_EXAMPLES": "ON" if self.options.with_examples else "OFF",
            "BUILD_APPS": "ON" if self.options.with_apps else "OFF",
        }
        cmake.configure(options)
        cmake.build()

    def package(self):
        # copy public headers from the source tree
        copy(
            self,
            pattern="*.h",
            src=os.path.join(self.source_folder, "src"),
            dst=os.path.join(self.package_folder, "include"),
            keep_path=True,
        )
        copy(
            self,
            pattern="*.hpp",
            src=os.path.join(self.source_folder, "src"),
            dst=os.path.join(self.package_folder, "include"),
            keep_path=True,
        )

        # copy libraries and binaries from the build folder
        # build outputs may end up in root build folder, lib/ or bin/ depending on platform
        lib_patterns = ("*.so*", "*.a", "*.lib", "*.dll", "*.dylib")
        for pattern in lib_patterns:
            # top-level build folder
            copy(
                self,
                pattern=pattern,
                src=os.path.join(self.build_folder, "lib"),
                dst=os.path.join(self.package_folder, "lib"),
                keep_path=True,
            )
            copy(
                self,
                pattern=pattern,
                src=os.path.join(self.build_folder, "bin"),
                dst=os.path.join(self.package_folder, "bin"),
                keep_path=True,
            )

        # copy any executables from bin/ folder
        copy(
            self,
            pattern="*",
            src=os.path.join(self.build_folder, "bin"),
            dst=os.path.join(self.package_folder, "bin"),
            keep_path=True,
        )
        # remove any cmake package config that might have been generated during build to avoid leaking buildsystem files
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.libs = ["f8studio"]
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.bindirs = ["bin"]

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy, download
from conan.tools.layout import basic_layout
from conan.tools.system.package_manager import Apt
import os


class LibMPV(ConanFile):
    name = "libmpv"
    version = "20251124"
    settings = "os", "arch", "compiler", "build_type"
    description = "Wrapper for libmpv pre-built binaries"
    license = "GPL-2.0-or-later"
    url = "https://github.com/mpv-player/mpv"

    _windows_archive = "mpv-dev-x86_64-v3-20251124-git-8469605.7z"
    _windows_url = (
        "https://github.com/shinchiro/mpv-winbuild-cmake/releases/download/"
        "20251124/mpv-dev-x86_64-v3-20251124-git-8469605.7z"
    )
    _windows_sha256 = "0484a25e0a750e0807583f97f128c8a37b4e901267806f3050417ebef6a307d4"

    def layout(self):
        basic_layout(self)

    def package_id(self):
        # Only the OS/arch matters; libmpv binaries are prebuilt
        self.info.settings.compiler = "Any"
        self.info.settings.build_type = "Any"

    def build_requirements(self):
        if self.settings.os == "Windows":
            self.tool_requires("7zip/25.01")

    def system_requirements(self):
        if self.settings.os == "Linux":
            apt = Apt(self)
            apt.install(["libmpv-dev"], update=True)

    def build(self):
        if self.settings.os == "Windows":
            if self.settings.arch != "x86_64":
                raise ConanInvalidConfiguration("Only x86_64 libmpv binaries are available for Windows.")
            archive = os.path.join(self.build_folder, self._windows_archive)
            download(self, self._windows_url, archive, sha256=self._windows_sha256)
            self.run(f"7z x \"{archive}\" -y", env="conanbuild")
            os.remove(archive)
        elif self.settings.os == "Linux":
            self.output.info("Using system-provided libmpv (libmpv-dev). Nothing to build.")
        else:
            raise ConanInvalidConfiguration("libmpv recipe currently supports Windows and Debian-based Linux.")

    def package(self):
        if self.settings.os == "Windows":
            root_dir = os.path.join(self.build_folder)
            include_src = os.path.join(root_dir, "include")
            lib_src = os.path.join(root_dir)
            bin_src = os.path.join(root_dir)

            copy(self, "*.h", src=include_src, dst=os.path.join(self.package_folder, "include"), keep_path=True)
            copy(self, "*.hpp", src=include_src, dst=os.path.join(self.package_folder, "include"), keep_path=True)
            copy(self, "*.lib", src=lib_src, dst=os.path.join(self.package_folder, "lib"), keep_path=True)
            copy(self, "*.dll.a", src=lib_src, dst=os.path.join(self.package_folder, "lib"), keep_path=True)
            copy(self, "*.dll", src=bin_src, dst=os.path.join(self.package_folder, "bin"), keep_path=True)
            copy(self, "*.pdb", src=bin_src, dst=os.path.join(self.package_folder, "bin"), keep_path=True)
        elif self.settings.os == "Linux":
            # When installing via the system package manager, headers and libs live under /usr.
            # No files need to be copied into the Conan package.
            pass
        else:
            raise ConanInvalidConfiguration("libmpv recipe currently supports Windows and Debian-based Linux.")

    def package_info(self):
        if self.settings.os == "Windows":
            self.cpp_info.includedirs = ["include"]
            self.cpp_info.libdirs = ["lib"]
            self.cpp_info.bindirs = ["bin"]
            self.cpp_info.libs = ["libmpv.dll.a"]

            bin_path = os.path.join(self.package_folder, "bin")
            self.runenv_info.append_path("PATH", bin_path)
        elif self.settings.os == "Linux":
            self.cpp_info.includedirs = []
            self.cpp_info.libdirs = []
            self.cpp_info.bindirs = []
            self.cpp_info.libs = []
            self.cpp_info.system_libs = ["mpv"]
        else:
            raise ConanInvalidConfiguration("libmpv recipe currently supports Windows and Debian-based Linux.")

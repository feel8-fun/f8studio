from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy, download
from conan.tools.layout import basic_layout
from conan.tools.system.package_manager import Apt
import os


class YtDlp(ConanFile):
    name = "ytdlp"
    version = "2025.11.12"
    settings = "os", "arch"
    description = "Packaged yt-dlp standalone executable"
    license = "Unlicense"
    url = "https://github.com/yt-dlp/yt-dlp"
    package_type = "application"
    short_paths = True

    _windows_filename = "yt-dlp.exe"
    _windows_url = "https://github.com/yt-dlp/yt-dlp/releases/download/2025.11.12/yt-dlp.exe"
    _windows_sha256 = "9f8b03a37125854895a7eebf50a605e34e7ec3bd2444931eff377f3ccec50e96"

    def layout(self):
        basic_layout(self)

    def system_requirements(self):
        if self.settings.os == "Linux":
            apt = Apt(self)
            apt.install(["yt-dlp"], update=True)

    def build(self):
        if self.settings.os == "Windows":
            if self.settings.arch != "x86_64":
                raise ConanInvalidConfiguration("yt-dlp recipe currently supports x86_64 Windows only.")
            dst = os.path.join(self.build_folder, self._windows_filename)
            download(self, self._windows_url, dst, sha256=self._windows_sha256)
        elif self.settings.os == "Linux":
            self.output.info("Using system-provided yt-dlp. Nothing to build.")
        else:
            raise ConanInvalidConfiguration("yt-dlp recipe currently supports Windows and Debian-based Linux.")

    def package(self):
        if self.settings.os == "Windows":
            copy(self, self._windows_filename, src=self.build_folder,
                 dst=os.path.join(self.package_folder, "bin"), keep_path=False)

    def package_info(self):
        if self.settings.os == "Windows":
            self.cpp_info.includedirs = []
            self.cpp_info.libdirs = []
            self.cpp_info.bindirs = ["bin"]
            self.runenv_info.append_path("PATH", os.path.join(self.package_folder, "bin"))
        else:
            self.cpp_info.includedirs = []
            self.cpp_info.libdirs = []
            self.cpp_info.bindirs = []

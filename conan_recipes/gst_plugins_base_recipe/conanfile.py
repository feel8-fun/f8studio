from conan import ConanFile
from conan.tools.env import VirtualBuildEnv, VirtualRunEnv
from conan.tools.files import copy, get
from conan.tools.gnu import PkgConfigDeps
from conan.tools.layout import basic_layout
from conan.tools.meson import Meson, MesonToolchain
from conan.tools.microsoft import is_msvc, check_min_vs

import glob
import os


class GstPluginsBaseConan(ConanFile):
    name = "gst-plugins-base"
    version = "1.24.7"
    license = "LGPL-2.0-or-later"
    url = "https://gstreamer.freedesktop.org/"
    description = "GStreamer Base Plugins and utility libraries (pbutils/app/audio/video)"
    topics = ("gstreamer", "plugins", "multimedia")
    package_type = "library"

    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {
        "shared": True,
        "fPIC": True,
        "gstreamer/*:shared": True,
        "glib/*:shared": True,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.get_safe("shared"):
            self.options.rm_safe("fPIC")
        self.settings.rm_safe("compiler.libcxx")
        self.settings.rm_safe("compiler.cppstd")

    def layout(self):
        basic_layout(self, src_folder="src")

    def requirements(self):
        self.requires(f"gstreamer/{self.version}")
        self.requires("zlib/1.3.1")

    def build_requirements(self):
        self.tool_requires("meson/[>=1.2.3 <2]")
        self.tool_requires("ninja/[>=1.10.2 <2]")
        if not self.conf.get("tools.gnu:pkg_config", check_type=str):
            self.tool_requires("pkgconf/[>=2.2 <3]")
        if self.settings.os == "Windows":
            self.tool_requires("winflexbison/2.5.25")

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        VirtualBuildEnv(self).generate()
        VirtualRunEnv(self).generate(scope="build")
        PkgConfigDeps(self).generate()

        tc = MesonToolchain(self)
        if is_msvc(self) and not check_min_vs(self, "190", raise_invalid=False):
            tc.project_options["c_std"] = "c99"

        tc.project_options["tests"] = "disabled"
        tc.project_options["examples"] = "disabled"
        tc.project_options["introspection"] = "disabled"
        tc.project_options["doc"] = "disabled"
        tc.project_options["tools"] = "disabled"

        # Only keep what we need for the WebRTC gateway probe (appsrc/appsink, videoconvert, etc.).
        tc.project_options["app"] = "enabled"
        tc.project_options["orc"] = "disabled"
        tc.project_options["gl"] = "disabled"
        tc.project_options["x11"] = "disabled"
        tc.project_options["alsa"] = "disabled"
        tc.project_options["ogg"] = "disabled"
        tc.project_options["theora"] = "disabled"
        tc.project_options["vorbis"] = "disabled"
        tc.project_options["pango"] = "disabled"

        tc.generate()

    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def _fix_library_names(self, path: str):
        if not is_msvc(self):
            return
        if not os.path.isdir(path):
            return
        for filename_old in glob.glob(os.path.join(path, "*.a")):
            base = os.path.basename(filename_old)
            if base.startswith("lib") and base.endswith(".a"):
                filename_new = os.path.join(path, base[3:-2] + ".lib")
                os.replace(filename_old, filename_new)

    def package(self):
        copy(self, "COPYING*", self.source_folder, os.path.join(self.package_folder, "licenses"))
        meson = Meson(self)
        meson.install()
        self._fix_library_names(os.path.join(self.package_folder, "lib"))

    def package_info(self):
        plugin_dir = os.path.join(self.package_folder, "lib", "gstreamer-1.0")
        self.runenv_info.append_path("GST_PLUGIN_PATH", plugin_dir)

        incdir = os.path.join(self.package_folder, "include", "gstreamer-1.0")
        incs = [os.path.join("include", "gstreamer-1.0")] if os.path.isdir(incdir) else []

        self.cpp_info.includedirs = incs
        self.cpp_info.libdirs = ["lib"]

        # Provide pkg-config names expected by other GStreamer modules.
        for name, libs, requires in [
            ("gstreamer-app-1.0", ["gstapp-1.0"], ["gstreamer::gstreamer-1.0"]),
            ("gstreamer-audio-1.0", ["gstaudio-1.0"], ["gstreamer::gstreamer-base-1.0"]),
            ("gstreamer-allocators-1.0", ["gstallocators-1.0"], ["gstreamer::gstreamer-base-1.0"]),
            ("gstreamer-fft-1.0", ["gstfft-1.0"], ["gstreamer::gstreamer-base-1.0"]),
            ("gstreamer-video-1.0", ["gstvideo-1.0"], ["gstreamer::gstreamer-base-1.0"]),
            (
                "gstreamer-rtp-1.0",
                ["gstrtp-1.0"],
                ["gstreamer::gstreamer-base-1.0", "gstreamer-audio-1.0", "gstreamer-video-1.0"],
            ),
            ("gstreamer-rtsp-1.0", ["gstrtsp-1.0"], ["gstreamer::gstreamer-base-1.0", "gstreamer-rtp-1.0"]),
            ("gstreamer-sdp-1.0", ["gstsdp-1.0"], ["gstreamer::gstreamer-base-1.0"]),
            ("gstreamer-pbutils-1.0", ["gstpbutils-1.0"], ["gstreamer::gstreamer-base-1.0"]),
            ("gstreamer-tag-1.0", ["gsttag-1.0"], ["gstreamer::gstreamer-base-1.0", "zlib::zlib"]),
            ("gstreamer-riff-1.0", ["gstriff-1.0"], ["gstreamer::gstreamer-base-1.0"]),
        ]:
            self.cpp_info.components[name].set_property("pkg_config_name", name)
            self.cpp_info.components[name].names["pkg_config"] = name
            self.cpp_info.components[name].libs = libs
            self.cpp_info.components[name].requires = requires
            self.cpp_info.components[name].includedirs = incs

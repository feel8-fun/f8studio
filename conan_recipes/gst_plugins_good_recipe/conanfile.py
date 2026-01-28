from conan import ConanFile
from conan.tools.env import VirtualBuildEnv, VirtualRunEnv
from conan.tools.files import copy, get
from conan.tools.gnu import PkgConfigDeps
from conan.tools.layout import basic_layout
from conan.tools.meson import Meson, MesonToolchain
from conan.tools.microsoft import is_msvc, check_min_vs

import os


class GstPluginsGoodConan(ConanFile):
    name = "gst-plugins-good"
    version = "1.24.7"
    license = "LGPL-2.0-or-later"
    url = "https://gstreamer.freedesktop.org/"
    description = "GStreamer Good Plugins (RTP/UDP depay/payloaders, etc.)"
    topics = ("gstreamer", "plugins", "rtp", "udp")
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
        self.requires(f"gst-plugins-base/{self.version}")
        self.requires("zlib/1.3.1")
        self.requires("libvpx/1.15.2")

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
        tc.project_options["doc"] = "disabled"

        # Keep only RTP/UDP pieces for depay/payload.
        tc.project_options["rtp"] = "enabled"
        tc.project_options["udp"] = "enabled"
        # VP8/VP9 decode/encode elements (vp8dec/vp9dec, etc.).
        tc.project_options["vpx"] = "enabled"
        tc.project_options["rtsp"] = "disabled"
        tc.project_options["v4l2"] = "disabled"
        tc.project_options["jpeg"] = "disabled"
        tc.project_options["png"] = "disabled"
        tc.project_options["soup"] = "disabled"

        tc.generate()

    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def package(self):
        copy(self, "COPYING*", self.source_folder, os.path.join(self.package_folder, "licenses"))
        meson = Meson(self)
        meson.install()

    def package_info(self):
        plugin_dir = os.path.join(self.package_folder, "lib", "gstreamer-1.0")
        self.runenv_info.append_path("GST_PLUGIN_PATH", plugin_dir)
        incdir = os.path.join(self.package_folder, "include", "gstreamer-1.0")
        self.cpp_info.includedirs = [os.path.join("include", "gstreamer-1.0")] if os.path.isdir(incdir) else []
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.requires = [
            "gstreamer::gstreamer-1.0",
            "gst-plugins-base::gstreamer-rtp-1.0",
            "libvpx::libvpx",
            "zlib::zlib",
        ]

from conan import ConanFile
from conan.tools.env import VirtualBuildEnv, VirtualRunEnv
from conan.tools.files import copy, get
from conan.tools.gnu import PkgConfigDeps
from conan.tools.layout import basic_layout
from conan.tools.meson import Meson, MesonToolchain
from conan.tools.microsoft import is_msvc, check_min_vs
from conan.tools.scm import Version

import glob
import os
import shutil


class GstPluginsBadConan(ConanFile):
    name = "gst-plugins-bad"
    version = "1.24.7"
    license = "LGPL-2.0-or-later"
    url = "https://gstreamer.freedesktop.org/"
    description = "GStreamer Bad Plugins (WebRTC, SRTP, DTLS, OpenH264, etc.)"
    topics = ("gstreamer", "webrtc", "rtp", "h264", "vp8", "multimedia")
    package_type = "library"

    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {
        "shared": True,
        "fPIC": True,
        # We want dynamic plugins on Windows, so build shared GStreamer + GLib.
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
        self.requires("openssl/3.6.0")
        self.requires("libsrtp/2.6.0")
        self.requires("libnice/0.1.21")
        self.requires("usrsctp/0.9.5.0")
        self.requires("openh264/2.6.0")
        self.requires("zlib/1.3.1")

    def build_requirements(self):
        self.tool_requires("meson/[>=1.2.3 <2]")
        self.tool_requires("ninja/[>=1.10.2 <2]")
        if not self.conf.get("tools.gnu:pkg_config", check_type=str):
            self.tool_requires("pkgconf/[>=2.2 <3]")
        if self.settings.os == "Windows":
            self.tool_requires("winflexbison/2.5.25")
        else:
            self.tool_requires("bison/3.8.2")
            self.tool_requires("flex/2.6.4")

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        VirtualBuildEnv(self).generate()
        VirtualRunEnv(self).generate(scope="build")

        PkgConfigDeps(self).generate()
        # Meson expects `dependency('nice')` but ConanCenter libnice doesn't declare a pkg-config name.
        # PkgConfigDeps generates `libnice.pc`, so provide an alias `nice.pc`.
        pc_dir = self.generators_folder
        libnice_pc = os.path.join(pc_dir, "libnice.pc")
        nice_pc = os.path.join(pc_dir, "nice.pc")
        if os.path.isfile(libnice_pc) and not os.path.isfile(nice_pc):
            shutil.copyfile(libnice_pc, nice_pc)
            try:
                with open(nice_pc, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("Cflags:"):
                        if "/nice" not in line and "\\nice" not in line:
                            lines[i] = line + ' -I"${includedir}/nice"'
                        break
                with open(nice_pc, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            except Exception:
                # Best-effort; if this fails, Meson will report missing headers.
                pass

        tc = MesonToolchain(self)
        if is_msvc(self) and not check_min_vs(self, "190", raise_invalid=False):
            tc.project_options["c_std"] = "c99"

        # Disable all "auto" features by default, then opt-in only to what we need.
        tc.project_options["auto_features"] = "disabled"

        # Keep the build minimal and deterministic.
        tc.project_options["tests"] = "disabled"
        tc.project_options["examples"] = "disabled"
        tc.project_options["introspection"] = "disabled"
        tc.project_options["doc"] = "disabled"
        tc.project_options["tools"] = "disabled"

        # Explicitly enable what we care about for the gateway.
        tc.project_options["webrtc"] = "enabled"
        tc.project_options["sctp"] = "enabled"
        tc.project_options["dtls"] = "enabled"
        tc.project_options["srtp"] = "enabled"
        tc.project_options["openh264"] = "enabled"
        tc.project_options["videoparsers"] = "enabled"

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
            # Some Meson builds still output .a on MSVC; rename to .lib.
            base = os.path.basename(filename_old)
            if base.startswith("lib") and base.endswith(".a"):
                filename_new = os.path.join(path, base[3:-2] + ".lib")
                os.replace(filename_old, filename_new)

    def package(self):
        copy(self, "COPYING*", self.source_folder, os.path.join(self.package_folder, "licenses"))
        meson = Meson(self)
        meson.install()

        self._fix_library_names(os.path.join(self.package_folder, "lib"))
        self._fix_library_names(os.path.join(self.package_folder, "lib", "gstreamer-1.0"))

    def package_info(self):
        # This package is primarily consumed at runtime via GStreamer plugin discovery.
        plugin_dir = os.path.join(self.package_folder, "lib", "gstreamer-1.0")
        self.runenv_info.append_path("GST_PLUGIN_PATH", plugin_dir)
        # webrtcbin depends on nicesrc/nicesink from libnice's GStreamer plugin.
        try:
            libnice_pkg = self.dependencies["libnice"].package_folder
            libnice_plugin_dir = os.path.join(libnice_pkg, "lib", "gstreamer-1.0")
            if os.path.isdir(libnice_plugin_dir):
                self.runenv_info.append_path("GST_PLUGIN_PATH", libnice_plugin_dir)
        except Exception:
            pass

        # Headers are optional; some builds install only plugins + helper libs.
        incdir = os.path.join(self.package_folder, "include", "gstreamer-1.0")
        self.cpp_info.includedirs = [os.path.join("include", "gstreamer-1.0")] if os.path.isdir(incdir) else []
        self.cpp_info.libdirs = ["lib"]

        # Linkable helper libraries that applications may need when driving webrtcbin directly.
        # (The actual elements are still loaded as plugins at runtime.)
        self.cpp_info.libs = [
            "gstwebrtc-1.0",
            "gstwebrtcnice-1.0",
            "gstsctp-1.0",
            "gstcodecparsers-1.0",
        ]
        self.cpp_info.requires = [
            "gstreamer::gstreamer-1.0",
            "gstreamer::gstreamer-base-1.0",
            "gst-plugins-base::gstreamer-sdp-1.0",
            "gst-plugins-base::gstreamer-rtp-1.0",
            "libnice::libnice",
            "libsrtp::libsrtp",
            "usrsctp::usrsctp",
            "openssl::openssl",
            "openh264::openh264",
            "zlib::zlib",
        ]

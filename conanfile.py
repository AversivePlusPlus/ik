from conans import ConanFile, CMake

class AversivePlusPlusModuleConan(ConanFile):
    name = "ik"
    version = "0.1"
    exports = "*.hpp"
    requires = "cas/0.1@AversivePlusPlus/dev"

    def package(self):
        self.copy("*.hpp", src="include", dst="include")

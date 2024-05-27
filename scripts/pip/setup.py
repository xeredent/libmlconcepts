import sys
import pathlib
import shutil
import abc
import os

from pathlib import Path
from shutil import copy2
from typing import Generator, List

import ninja
import cmake
import cmake_build_extension
import setuptools
import setuptools.command.sdist
import setuptools_scm



class CustomSDist(abc.ABC, setuptools.command.sdist.sdist):
    
    @staticmethod
    def get_sdist_files(repo_root: str) -> List[Path]:
        srcfiles = [f for f in (Path(repo_root) / "src").glob(pattern="**/*")
                    if not f.is_dir()]
        incfiles = [f for f in (Path(repo_root) / "include").glob(pattern="**/*")
                    if not f.is_dir()]
        testfiles = [f for f in (Path(repo_root) / "tests").glob(pattern="**/*")
                    if not f.is_dir()]

        return [Path(repo_root) / "CMakeLists.txt"] + list(srcfiles) + list(incfiles) + list(testfiles)

    def make_release_tree(self, base_dir, files) -> None:
        # Build the setuptools_scm configuration, containing useful info for the sdist
        config: setuptools_scm.Configuration = (
            setuptools_scm.Configuration.from_file(
                dist_name=self.distribution.metadata.name
            )
        )

        # Get the root of the git repository
        repo_root = config.absolute_root

        if not Path(repo_root).exists() or not Path(repo_root).is_dir():
            raise RuntimeError(f"Failed to find a git repo in {repo_root}")

        # Prepare the release tree by calling the original method
        super(CustomSDist, self).make_release_tree(base_dir=base_dir, files=files)

        # Collect all the files and copy them in the subfolder containing setup.cfg
        for file in self.get_sdist_files(repo_root=repo_root):

            src = Path(file)
            dst = Path(base_dir) / Path(file).relative_to(repo_root)

            # Make sure that the parent directory of the destination exists
            dst.absolute().parent.mkdir(parents=True, exist_ok=True)

            print(f"{Path(file).relative_to(repo_root)} -> {dst}")
            copy2(src=src, dst=dst)

        # Create the updated list of files included in the sdist
        all_files_gen = Path(base_dir).glob(pattern="**/*")
        all_files = [str(f.relative_to(base_dir)) for f in all_files_gen]

        # Find the SOURCES.txt file
        sources_txt_list = list(Path(base_dir).glob(pattern=f"*.egg-info/SOURCES.txt"))
        assert len(sources_txt_list) == 1

        # Update the SOURCES.txt files with the real content of the sdist
        os.unlink(sources_txt_list[0])
        with open(file=sources_txt_list[0], mode="w") as f:
            f.write("\n".join([str(f) for f in all_files]))


if (Path(".") / "CMakeLists.txt").exists():
    # Install from sdist
    source_dir = str(Path(".").absolute())
else:
    # Install from sources or build wheel
    source_dir = str(Path(".").absolute().parent.parent)

#Prioritize the cmake bin from cmake's installation
if os.path.isdir(cmake.CMAKE_BIN_DIR):
    os.environ["PATH"] = cmake.CMAKE_BIN_DIR + os.pathsep + os.environ["PATH"]
    
if os.path.isdir(ninja.BIN_DIR):
    sys.path.insert(0, ninja.BIN_DIR)
    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ["PATH"]

cmake_options = []
cmake_options.append("-DPYTHON_EXECUTABLE={}".format(sys.executable))
cmake_options.append("-DPython_EXECUTABLE={}".format(sys.executable))

setuptools.setup(
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="CMakeProject",
            install_prefix="mlconcepts",
            cmake_depends_on=["pybind11"],
            disable_editable=True,
            cmake_configure_options=cmake_options,
            source_dir = source_dir,
            cmake_build_type = "Release"
        )
    ],
    cmdclass=dict(
            build_ext=cmake_build_extension.BuildExtension,
            sdist=CustomSDist
    ),
)
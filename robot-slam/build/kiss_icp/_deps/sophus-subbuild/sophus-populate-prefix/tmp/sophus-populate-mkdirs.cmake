# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-src")
  file(MAKE_DIRECTORY "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-src")
endif()
file(MAKE_DIRECTORY
  "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-build"
  "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix"
  "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix/tmp"
  "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix/src/sophus-populate-stamp"
  "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix/src"
  "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix/src/sophus-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix/src/sophus-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/build/kiss_icp/_deps/sophus-subbuild/sophus-populate-prefix/src/sophus-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

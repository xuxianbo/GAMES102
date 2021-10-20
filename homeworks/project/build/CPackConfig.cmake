# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "E:/Games/102/Study/homeworks/project;E:/Games/102/Study/homeworks/project/build")
set(CPACK_CMAKE_GENERATOR "Visual Studio 16 2019")
set(CPACK_COMPONENTS_ALL "Runtime;Library;Header;Data;Documentation;Example;Other")
set(CPACK_COMPONENTS_ALL_SET_BY_USER "TRUE")
set(CPACK_COMPONENT_DATA_DESCRIPTION "Application data. Installed into share/lua.")
set(CPACK_COMPONENT_DATA_DISPLAY_NAME "lua Data")
set(CPACK_COMPONENT_DOCUMENTATION_DESCRIPTION "Application documentation. Installed into share/lua/doc.")
set(CPACK_COMPONENT_DOCUMENTATION_DISPLAY_NAME "lua Documentation")
set(CPACK_COMPONENT_EXAMPLE_DESCRIPTION "Examples and their associated data. Installed into share/lua/example.")
set(CPACK_COMPONENT_EXAMPLE_DISPLAY_NAME "lua Examples")
set(CPACK_COMPONENT_HEADER_DESCRIPTION "Headers needed for development. Installed into include.")
set(CPACK_COMPONENT_HEADER_DISPLAY_NAME "lua Development Headers")
set(CPACK_COMPONENT_LIBRARY_DESCRIPTION "Static and import libraries needed for development. Installed into lib or bin.")
set(CPACK_COMPONENT_LIBRARY_DISPLAY_NAME "lua Development Libraries")
set(CPACK_COMPONENT_OTHER_DESCRIPTION "Other unspecified content. Installed into share/lua/etc.")
set(CPACK_COMPONENT_OTHER_DISPLAY_NAME "lua Unspecified Content")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Executables and runtime libraries. Installed into bin.")
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "lua Runtime")
set(CPACK_COMPONENT_TEST_DESCRIPTION "Tests and associated data. Installed into share/lua/test.")
set(CPACK_COMPONENT_TEST_DISPLAY_NAME "lua Tests")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "C:/Software/CMake/share/cmake-3.22/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "GAMES102_Project built using CMake")
set(CPACK_GENERATOR "ZIP")
set(CPACK_INSTALL_CMAKE_PROJECTS "E:/Games/102/Study/homeworks/project/build;GAMES102_Project;ALL;/")
set(CPACK_INSTALL_PREFIX "C:/Program Files (x86)/Ubpa")
set(CPACK_MODULE_PATH "E:/Games/102/Study/homeworks/project/build/_deps/ulua-src/cmake")
set(CPACK_NSIS_DISPLAY_NAME "lua 5.3.2")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "lua 5.3.2")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OUTPUT_CONFIG_FILE "E:/Games/102/Study/homeworks/project/build/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "C:/Software/CMake/share/cmake-3.22/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GAMES102_Project built using CMake")
set(CPACK_PACKAGE_FILE_NAME "lua-5.3.2-win32")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "lua 5.3.2")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "lua 5.3.2")
set(CPACK_PACKAGE_NAME "lua")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "LuaDist")
set(CPACK_PACKAGE_VERSION "5.3.2")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "2")
set(CPACK_RESOURCE_FILE_LICENSE "C:/Software/CMake/share/cmake-3.22/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "C:/Software/CMake/share/cmake-3.22/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "C:/Software/CMake/share/cmake-3.22/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_7Z "ON")
set(CPACK_SOURCE_GENERATOR "7Z;ZIP")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "E:/Games/102/Study/homeworks/project/build/CPackSourceConfig.cmake")
set(CPACK_SOURCE_ZIP "ON")
set(CPACK_STRIP_FILES "TRUE")
set(CPACK_SYSTEM_NAME "win32")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "win32")
set(CPACK_WIX_SIZEOF_VOID_P "4")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "E:/Games/102/Study/homeworks/project/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()

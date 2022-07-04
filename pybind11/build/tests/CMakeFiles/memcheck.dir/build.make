# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build

# Utility rule file for memcheck.

# Include the progress variables for this target.
include tests/CMakeFiles/memcheck.dir/progress.make

tests/CMakeFiles/memcheck: tests/pybind11_tests.cpython-37m-x86_64-linux-gnu.so
tests/CMakeFiles/memcheck: tests/pybind11_cross_module_tests.cpython-37m-x86_64-linux-gnu.so
tests/CMakeFiles/memcheck: tests/cross_module_gil_utils.cpython-37m-x86_64-linux-gnu.so
	cd /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests && PYTHONMALLOC=malloc valgrind --leak-check=full --show-leak-kinds=definite,indirect --errors-for-leak-kinds=definite,indirect --error-exitcode=1 --read-var-info=yes --track-origins=yes --suppressions="/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/valgrind-python.supp" --suppressions="/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/valgrind-numpy-scipy.supp" --gen-suppressions=all /home/h2amer/anaconda3/bin/python -m pytest /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_async.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_buffers.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_builtin_casters.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_call_policies.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_callbacks.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_chrono.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_class.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_const_name.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_constants_and_functions.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_copy_move.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_custom_type_casters.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_custom_type_setup.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_docstring_options.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_eigen.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_enum.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_eval.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_exceptions.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_factory_constructors.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_gil_scoped.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_iostream.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_kwargs_and_defaults.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_local_bindings.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_methods_and_attributes.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_modules.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_multiple_inheritance.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_numpy_array.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_numpy_dtypes.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_numpy_vectorize.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_opaque_types.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_operator_overloading.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_pickling.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_pytypes.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_sequences_and_iterators.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_smart_ptr.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_stl.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_stl_binders.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_tagbased_polymorphic.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_thread.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_union.py /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests/test_virtual_functions.py

memcheck: tests/CMakeFiles/memcheck
memcheck: tests/CMakeFiles/memcheck.dir/build.make

.PHONY : memcheck

# Rule to build all files generated by this target.
tests/CMakeFiles/memcheck.dir/build: memcheck

.PHONY : tests/CMakeFiles/memcheck.dir/build

tests/CMakeFiles/memcheck.dir/clean:
	cd /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/memcheck.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/memcheck.dir/clean

tests/CMakeFiles/memcheck.dir/depend:
	cd /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11 /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/tests /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests /home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/pybind11/build/tests/CMakeFiles/memcheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/memcheck.dir/depend


from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, IS_WINDOWS, build_ext
import copy
import glob
import os.path as osp


# compile utils
def MPSExtension(name, sources, *args, **kwargs):
    from torch.utils.cpp_extension import CppExtension

    extra_compile_args = {}
    extra_compile_args['cxx'] = ['-Wall', '-std=c++17']
    extra_compile_args['cxx'] += [
        '-framework', 'Metal', '-framework', 'Foundation'
    ]
    extra_compile_args['cxx'] += ['-ObjC++']

    kwargs['extra_compile_args'] = kwargs.get('extra_compile_args', {})
    kwargs['extra_compile_args'].update(extra_compile_args)
    return CppExtension(name, sources, *args, **kwargs)


def MetalExtension(name, sources, *args, **kwargs):
    from setuptools import Extension
    for src in sources:
        assert osp.splitext(
            src)[1] == '.metal', f'Expect .metal file, but get {src}.'

    return Extension(name, sources, *args, **kwargs)


def _is_metal_file(path: str) -> bool:
    return osp.splitext(path)[1] == '.metal'


class BuildMPSExtension(BuildExtension):

    def build_extensions(self) -> None:
        self._check_abi()

        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'objc', 'metal']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []
            self._add_compile_flag(extension,
                                   '-DTORCH_API_INCLUDE_EXTENSION_H')

            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension,
                                           f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # register .mm .metal as valid type
        self.compiler.src_extensions += ['.mm', '.metal']
        original_compile = self.compiler._compile
        original_link = self.compiler.link
        original_object_filenames = self.compiler.object_filenames

        def darwin_wrap_single_compile(obj, src, ext, cc_args, extra_postargs,
                                       pp_opts) -> None:
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so

                if _is_metal_file(src):
                    metal = ['xcrun', 'metal']
                    self.compiler.set_executable('compiler_so', metal)
                    if isinstance(cflags, dict):
                        cflags = cflags.get('metal', [])
                    else:
                        cflags = []
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                self.compiler.set_executable('compiler_so', original_compiler)

        def darwin_wrap_single_link(target_desc,
                                    objects,
                                    output_filename,
                                    output_dir=None,
                                    libraries=None,
                                    library_dirs=None,
                                    runtime_library_dirs=None,
                                    export_symbols=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    build_temp=None,
                                    target_lang=None):
            if osp.splitext(objects[0])[1].lower() == '.air':
                for obj in objects:
                    assert osp.splitext(obj)[1].lower(
                    ) == '.air', f'Expect .air file, but get {obj}.'

                linker = ['xcrun', 'metallib']
                self.compiler.spawn(linker + objects + ['-o', output_filename])
            else:
                return original_link(target_desc, objects, output_filename,
                                     output_dir, libraries, library_dirs,
                                     runtime_library_dirs, export_symbols,
                                     debug, extra_preargs, extra_postargs,
                                     build_temp, target_lang)

        def darwin_wrap_object_filenames(source_filenames,
                                         strip_dir=0,
                                         output_dir=''):
            src_name = source_filenames[0]
            old_obj_extension = self.compiler.obj_extension
            if osp.splitext(src_name)[1].lower() == '.metal':
                self.compiler.obj_extension = '.air'

            ret = original_object_filenames(source_filenames, strip_dir,
                                            output_dir)
            self.compiler.obj_extension = old_obj_extension

            return ret

        self.compiler._compile = darwin_wrap_single_compile
        self.compiler.link = darwin_wrap_single_link
        self.compiler.object_filenames = darwin_wrap_object_filenames
        build_ext.build_extensions(self)


def get_extensions():
    extensions = []

    # mps setting
    sources = glob.glob('./csrc/pytorch/*.cpp')
    sources += glob.glob('./csrc/pytorch/mps/*.mm')
    name = '_mps_test'

    ext = MPSExtension(name, sources)
    extensions.append(ext)

    # metal setting
    name = '_metal_kernel'
    sources = glob.glob('./csrc/pytorch/mps/*.metal')
    ext = MetalExtension(name, sources)
    extensions.append(ext)

    return extensions


cmd_class = {'build_ext': BuildMPSExtension}

if __name__ == '__main__':
    setup(
        name='mps_test',
        version='1.0',
        description='whatever',
        author='grimoire',
        ext_modules=get_extensions(),
        cmdclass=cmd_class)

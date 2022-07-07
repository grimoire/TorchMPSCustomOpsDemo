from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, IS_WINDOWS, build_ext
import copy
import glob
import os.path as osp


# compile utils
def MPSExtension(name, sources, *args, **kwargs):
    from torch.utils.cpp_extension import CppExtension

    # # add support to .mm
    # from distutils.unixccompiler import UnixCCompiler
    # if '.mm' not in UnixCCompiler.src_extensions:
    #     UnixCCompiler.src_extensions.append('.mm')
    #     UnixCCompiler.language_map['.mm'] = 'objc'

    extra_compile_args = {}
    extra_compile_args['cxx'] = ['-Wall', '-std=c++17']
    extra_compile_args['cxx'] += [
        '-framework', 'Metal', '-framework', 'Foundation'
    ]
    extra_compile_args['cxx'] += ['-ObjC++']

    kwargs['extra_compile_args'] = kwargs.get('extra_compile_args', {})
    kwargs['extra_compile_args'].update(extra_compile_args)
    return CppExtension(name, sources, *args, **kwargs)


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

        def darwin_wrap_single_compile(obj, src, ext, cc_args, extra_postargs,
                                       pp_opts) -> None:
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so

                if _is_metal_file(src):
                    metal = ['xcrun metal']
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

        self.compiler._compile = darwin_wrap_single_compile
        build_ext.build_extensions(self)


def get_extensions():
    extensions = []

    sources = glob.glob('./csrc/pytorch/*.cpp')
    sources += glob.glob('./csrc/pytorch/mps/*.mm')
    name = '_mps_test'

    ext = MPSExtension(name, sources)
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

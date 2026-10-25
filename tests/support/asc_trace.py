"""Owned native setup and host-only ASC trace shim; no device arithmetic or retirement proof."""

import shutil
import subprocess
from importlib import import_module

import pytest

from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.passes.lower_native import lower_native_structure


def memory(source):
    structure = lower_native_structure(source)
    return structure, collect_native_memory(structure.module)


def forbid_analysis(monkeypatch):

    def forbidden(*args, **kwargs):
        raise AssertionError("formatting must consume the existing owned facts")

    targets = (
        ("ascendflow.target.native", ("native_contract",)),
        ("ascendflow.passes.lower_native", ("lower_native_structure",)),
        ("ascendflow.analysis.structured_memory", ("collect_native_memory",)),
        ("ascendflow.analysis.structured_bounds", ("admit_memory_bounds",)),
        ("ascendflow.analysis.access_order", ("build_access_order",)),
    )
    modules = [(import_module(name), attributes) for name, attributes in targets]
    for module, attributes in modules:
        for attribute in attributes:
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, forbidden)


TRACE_SHIM = r"""
#include <cstdint>
#include <iostream>
#include <cstdlib>
#define __global__
#define __aicore__
#define __vector__
#define __cube__
#define __mix__(a,b)
#define __gm__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define ASC_IS_AIC (trace_actor == 0)
#define ASC_IS_AIV (trace_actor != 0)
using half=uint16_t; using event_t=int;
enum {PIPE_S,PIPE_MTE2,PIPE_MTE1,PIPE_M,PIPE_FIX,PIPE_V,PIPE_MTE3,inc};
int64_t trace_group=0,trace_actor=1;
void asc_init(){} uint64_t asc_get_ctrl(){return 0;} void asc_set_ctrl(uint64_t){}
int64_t asc_get_block_idx(){return trace_group;}
int64_t asc_get_sub_block_id(){return trace_actor-1;}
void asc_set_l0c_copy_params(int,int,int){}
void asc_sync_notify(int,int,int){} void asc_sync_wait(int,int,int){}
void asc_sync_pipe(int){}
template<class... P> void record(const char* name,float scalar,P... p){
 std::cout<<name<<" "<<scalar; ((std::cout<<" "<<uintptr_t(p)),...); std::cout<<"\n";
}
struct asc_load3d_v2_config{uint64_t config;};
struct asc_l13d_fmatrix_config{uint64_t config;};
void asc_set_l13d_rpt(asc_load3d_v2_config){record("set_load3d",0);}
void asc_set_l13d_padding(uint64_t){} void asc_set_l13d_size(uint64_t){}
void asc_set_l13d_fmatrix(asc_l13d_fmatrix_config){record("set_fmatrix_a",0);}
void asc_set_l13d_fmatrix_b(asc_l13d_fmatrix_config){record("set_fmatrix_b",0);}
template<class D,class S,class... P> void asc_copy_gm2ub(D d,S s,P...){record("copy",0,s,d);}
template<class D,class S,class... P> void asc_copy_ub2gm(D d,S s,P...){record("copy",0,s,d);}
template<class D,class S,class... P> void asc_copy_gm2l1_nd2nz(D d,S s,P...){record("nd2nz",0,s,d);}
template<class D,class S,class... P> void asc_copy_l12l0a(D d,S s,P...){
 record(sizeof...(P)>4?"load3d_a":"load2d_a",0,s,d);
}
template<class D,class S,class... P> void asc_copy_l12l0b(D d,S s,P...){
 record(sizeof...(P)>4?"load3d_b_nn":"load2d_b_nt",0,s,d);
}
template<class D,class S,class... P> void load_cbuf_to_cb(D d,S s,P...){record("load2d_b_nn",0,s,d);}
template<class D,class A,class B,class... P> void asc_mmad(D d,A a,B b,P...){record("mad",0,a,b,d);}
template<class D,class S,class... P> void asc_copy_l0c2gm(D d,S s,P...){record("fix",0,s,d);}
template<class D,class S> void asc_mul_scalar(D d,S s,float f,uint32_t){record("muls",f,s,d);}
template<class D> void asc_duplicate(D d,float f,uint32_t){record("fill",f,d);}
"""


def compile_cpp(tmp_path, source, *, sanitize=False):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required for scalar/address trace")
    path = tmp_path / "trace.cpp"
    path.write_text(source)
    binary = tmp_path / "trace"
    flags = ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"] if sanitize else []
    subprocess.run(
        [compiler, "-std=c++20", "-O0", *flags, str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return binary

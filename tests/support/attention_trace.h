// Test-only host instrumentation of the actual hand-written reference control.
// No arithmetic emulation or claim about instruction retirement within a pipe.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#define __aicore__
#define __global__
#define __gm__
#define __mix__(...)
inline int trace_actor=0, trace_group=0;
#define ASCEND_IS_AIC (trace_actor==0)
#define ASCEND_IS_AIV (trace_actor!=0)
using half=uint16_t;
using bfloat16_t=uint16_t;
using event_t=int;
enum pipe_t {PIPE_S,PIPE_V,PIPE_MTE1,PIPE_MTE2,PIPE_MTE3,PIPE_M,PIPE_FIX,PIPE_ALL};
inline const char* pipe_name(pipe_t p) { const char* n[]={"S","V","MTE1","MTE2","MTE3","M","FIX","ALL"};return n[p]; }
inline void trace(const char* action,pipe_t pipe,int id=-1,pipe_t dst=PIPE_S) {
 std::cout<<trace_actor<<' '<<action<<' '<<pipe_name(pipe)<<' '<<pipe_name(dst)<<' '<<id<<'\n';
}
inline int get_block_idx(){return trace_group;}
inline uint64_t get_ctrl(){return 0;}
inline void set_ctrl(uint64_t){}
inline void set_flag(pipe_t a,pipe_t b,int id){trace("set",a,id,b);}
inline void wait_flag(pipe_t a,pipe_t b,int id){trace("wait",b,id,a);}
namespace AscendC {
enum class HardEvent {MTE1_MTE2,M_MTE1,MTE2_MTE1,MTE1_M,FIX_M,M_FIX,V_MTE2,MTE2_V,MTE3_V,V_MTE3};
inline std::pair<pipe_t,pipe_t> bank(HardEvent e){
 switch(e){
#define BANK(a,b) case HardEvent::a##_##b:return {PIPE_##a,PIPE_##b};
 BANK(MTE1,MTE2) BANK(M,MTE1) BANK(MTE2,MTE1) BANK(MTE1,M) BANK(FIX,M) BANK(M,FIX)
 BANK(V,MTE2) BANK(MTE2,V) BANK(MTE3,V) BANK(V,MTE3)
#undef BANK
 } std::abort();
}
template<HardEvent E> void SetFlag(int id){auto [a,b]=bank(E);set_flag(a,b,id);}
template<HardEvent E> void WaitFlag(int id){auto [a,b]=bank(E);wait_flag(a,b,id);}
template<int MODE,pipe_t PIPE> void CrossCoreSetFlag(int id){static_assert(MODE==2);trace("send",PIPE,id);}
template<int MODE=0,pipe_t PIPE=PIPE_S> void CrossCoreWaitFlag(int id){static_assert(MODE==2);trace("recv",PIPE_S,id);}
template<pipe_t PIPE> void PipeBarrier(){trace(PIPE==PIPE_ALL?"drain":"op",PIPE);}
inline int GetSubBlockIdx(){return trace_actor-1;}
enum class TPosition {A1,A2,B2,CO1,VECCALC};
template<class T> struct LocalTensor {
 LocalTensor operator[](int64_t) const{return {};}
 template<class U> LocalTensor<U> ReinterpretCast()const{return {};}
};
template<class T> struct GlobalTensor {
 void SetGlobalBuffer(T*){}
 GlobalTensor operator[](int64_t)const{return {};}
};
template<TPosition P> struct TBuf{template<class T> LocalTensor<T> Get(){return {};}};
struct TPipe{
 TPipe(){if(trace_actor==0)for(int i=0;i<3;++i)SetFlag<HardEvent::M_MTE1>(i);}
 ~TPipe(){if(trace_actor==0)for(int i=0;i<3;++i)WaitFlag<HardEvent::M_MTE1>(i);}
 template<class T> void InitBuffer(T&,int){}
};
struct Nd2NzParams{int ndNum,nValue,dValue,srcNdMatrixStride,srcDValue,dstNzC0Stride,dstNzNStride,dstNzMatrixStride;};
struct LoadData2DParams{int startIndex,repeatTimes,srcStride,dstGap,sid;bool ifTranspose;};
struct MmadParams{int m,n,k,unitFlag;bool cmatrixInitVal;};
enum class QuantMode_t {NoQuant};
struct FixpipeParamsV220{int mSize,nSize,srcStride,dstStride,unitFlag;bool reluEn;QuantMode_t quantPre;};
constexpr int CFG_ROW_MAJOR=0;
inline void SetFixpipeNz2ndFlag(int,int,int){}
template<class T> void DataCopy(LocalTensor<T>,GlobalTensor<T>,Nd2NzParams){trace("op",PIPE_MTE2);}
template<class T> void DataCopy(LocalTensor<T>,GlobalTensor<T>,int){trace("op",PIPE_MTE2);}
template<class T> void DataCopy(GlobalTensor<T>,LocalTensor<T>,int){trace("op",PIPE_MTE3);}
template<class T> void LoadData(LocalTensor<T>,LocalTensor<T>,LoadData2DParams){trace("op",PIPE_MTE1);}
template<class T> void Mmad(LocalTensor<float>,LocalTensor<T>,LocalTensor<T>,MmadParams){trace("op",PIPE_M);}
template<class A,class B,int CFG> void Fixpipe(GlobalTensor<A>,LocalTensor<B>,FixpipeParamsV220){trace("op",PIPE_FIX);}
struct BinaryRepeatParams{BinaryRepeatParams(int,int,int,int,int,int){}};
struct BrcbRepeatParams{BrcbRepeatParams(int,int){}};
enum class ReduceType {MAX,SUM};
enum class ReduceOrder {ORDER_ONLY_VALUE};
enum class RoundMode {CAST_RINT};
template<ReduceType KIND,class... Args> void ReduceRepeat(Args...){trace("op",PIPE_V);}
#define VOP(name) template<class... Args> void name(Args...){trace("op",PIPE_V);}
VOP(Add) VOP(Sub) VOP(Mul) VOP(Muls) VOP(Max) VOP(Brcb) VOP(Adds) VOP(Exp) VOP(Div) VOP(Duplicate) VOP(Cast)
#undef VOP
}

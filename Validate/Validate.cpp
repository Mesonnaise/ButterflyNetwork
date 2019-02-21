#include<algorithm>
#include<chrono>
#include<cstring>
#include<iostream>
#include<random>

#include<immintrin.h>

#include"Networks.h"

template<typename Func>
__declspec(noinline) std::chrono::duration<double> ValidateBfly(
  Func op,size_t passes,size_t size,__m256i *value,__m256i *mask,__m256i *ret){

  const auto T1=std::chrono::high_resolution_clock::now();

  while(passes--){
    for(size_t idx=0;idx<size;idx++)
      *(ret+idx)=op(*(value+idx),*(mask+idx));
  }

  const auto T2=std::chrono::high_resolution_clock::now();

  return T2-T1;
}

__declspec(noinline) std::chrono::duration<double> BuiltinScatter64x4(
  size_t passes,size_t size,__m256i *value,__m256i *mask,__m256i *ret){

  uint64_t *m=reinterpret_cast<uint64_t*>(mask);
  uint64_t *v=reinterpret_cast<uint64_t*>(value);
  uint64_t *r=reinterpret_cast<uint64_t*>(ret);

  const auto T1=std::chrono::high_resolution_clock::now();

  while(passes--){
    for(size_t idx=0;idx<(size*4);idx++)
      *(r+idx)=_pdep_u64(*(v+idx),*(m+idx));
  }

  const auto T2=std::chrono::high_resolution_clock::now();

  return T2-T1;
}

__declspec(noinline) std::chrono::duration<double> BuiltinGather64x4(
  size_t passes,size_t size,__m256i *value,__m256i *mask,__m256i *ret){

  uint64_t *m=reinterpret_cast<uint64_t*>(mask);
  uint64_t *v=reinterpret_cast<uint64_t*>(value);
  uint64_t *r=reinterpret_cast<uint64_t*>(ret);

  const auto T1=std::chrono::high_resolution_clock::now();

  while(passes--){
    for(size_t idx=0;idx<(size*4);idx++)
      *(r+idx)=_pext_u64(*(v+idx),*(m+idx));
  }

  const auto T2=std::chrono::high_resolution_clock::now();

  return T2-T1;
}

__declspec(noinline) std::chrono::duration<double> BuiltinScatter256(
  size_t passes,size_t size,__m256i *value,__m256i *mask,__m256i *ret){

  const auto T1=std::chrono::high_resolution_clock::now();

  while(passes--){
    for(size_t idx=0;idx<size;idx++){
      uint64_t *m=reinterpret_cast<uint64_t*>(mask+idx);
      uint64_t *v=reinterpret_cast<uint64_t*>(value+idx);
      uint64_t *r=reinterpret_cast<uint64_t*>(ret+idx);

      uint64_t bitOff=0;

      for(size_t sub=0;sub<4;sub++){
        const uint64_t shift=bitOff%64;
        const uint64_t off  =bitOff/64;

        uint64_t lower=*(v+off);
        uint64_t upper=*(v+off+1);

        lower>>=shift;
        upper<<=64-shift;

        uint64_t comb=(shift==0?0:upper)|lower;

        *(r+sub)=_pdep_u64(comb,*(m+sub));
        bitOff +=_mm_popcnt_u64(*(m+sub));
      }
    }
  }

  const auto T2=std::chrono::high_resolution_clock::now();

  return T2-T1;
}

__declspec(noinline) std::chrono::duration<double> BuiltinGather256(
  size_t passes,size_t size,__m256i *value,__m256i *mask,__m256i *ret){

  //The algorithm for pext is adapted from a succinct array packer,
  //and requires the memory block to be zeroed out before use. 
  memset(ret,0,sizeof(__m256i)*size);

  const auto T1=std::chrono::high_resolution_clock::now();

  while(passes--){
    for(size_t idx=0;idx<size;idx++){
      uint64_t *m=reinterpret_cast<uint64_t*>(mask+idx);
      uint64_t *v=reinterpret_cast<uint64_t*>(value+idx);
      uint64_t *r=reinterpret_cast<uint64_t*>(ret+idx);

      uint64_t dstBitPos=0;
      uint64_t dstIdx   =0;

      for(uint64_t sub=0;sub<4;sub++){
        uint64_t packed=_pext_u64(*(v+sub),*(m+sub));
        uint64_t packedSize=_mm_popcnt_u64(*(m+sub));

        while(packedSize){
          *(r+dstIdx)&=~(0xFFFFFFFFFFFFFFFFULL<<dstBitPos);
          *(r+dstIdx)|=packed<<dstBitPos;

          uint64_t used=std::min(packedSize,64-dstBitPos);

          packedSize-=used;
          dstBitPos +=used;

          dstIdx+=dstBitPos/64;

          dstBitPos&=0x000000000000003FULL;
          packed>>=used;
        }
      }
    }
  }

  const auto T2=std::chrono::high_resolution_clock::now();

  return T2-T1;
}

void Match(size_t size,__m256i *ret1,__m256i *ret2){
  using namespace std;

  auto result=memcmp(ret1,ret2,sizeof(__m256i)*size);

  if(result==0)
    cout<<"AVX matches Native"<<endl;
  else
    cout<<"AVX does not match Native"<<endl;

  cout<<endl;
}

int main(){
  using namespace std;

  const size_t passes=2000;
  const size_t size=  1024;

  //BuiltinScatter256 cheats by reading one passed the end of the array,
  //instead of doing proper bounds checking.
  __m256i *val= reinterpret_cast<__m256i*>(_mm_malloc(sizeof(__m256i)*(size+1),32));
  __m256i *mask=reinterpret_cast<__m256i*>(_mm_malloc(sizeof(__m256i)*size,32));
  __m256i *ret1=reinterpret_cast<__m256i*>(_mm_malloc(sizeof(__m256i)*size,32));
  __m256i *ret2=reinterpret_cast<__m256i*>(_mm_malloc(sizeof(__m256i)*size,32));

  minstd_rand0 rand(std::random_device{}());
  uniform_int_distribution<uint64_t> dist;

  auto initializer=[&rand,&dist]()->__m256i{
    __m256i ret;

    ret.m256i_u64[0]=dist(rand);
    ret.m256i_u64[1]=dist(rand);
    ret.m256i_u64[2]=dist(rand);
    ret.m256i_u64[3]=dist(rand);

    return ret;
  };

  generate(val ,val+size ,initializer);
  generate(mask,mask+size,initializer);

  chrono::duration<double> delta;

  //Scatter(pdep) 64bits by 4

  delta=ValidateBfly(
    Butterfly::Scatter64x4,
    passes,size,val,mask,ret1);

  cout<<"AVX    Scatter64x4: "<<delta.count()<<endl;

  delta=BuiltinScatter64x4(passes,size,val,mask,ret2);

  cout<<"Native Scatter64x4: "<<delta.count()<<endl;

  Match(size,ret1,ret2);

  //Gather(pext) 64bits by 4

  delta=ValidateBfly(
    Butterfly::Gather64x4,
    passes,size,val,mask,ret1);

  cout<<"AVX    Gather64x4: "<<delta.count()<<std::endl;

  delta=BuiltinGather64x4(passes,size,val,mask,ret2);

  cout<<"Native Gather64x4: "<<delta.count()<<std::endl;

  Match(size,ret1,ret2);

  //Scatter(pdep) 256bits

  delta=ValidateBfly(
    Butterfly::Scatter256,
    passes,size,val,mask,ret1);

  std::cout<<"AVX    Scatter256: "<<delta.count()<<std::endl;

  delta=BuiltinScatter256(passes,size,val,mask,ret2);

  std::cout<<"Native Scatter256: "<<delta.count()<<std::endl;

  Match(size,ret1,ret2);

  //Gather(pext) 256bits

  delta=ValidateBfly(
    Butterfly::Gather256,
    passes,size,val,mask,ret1);

  std::cout<<"AVX    Gather256: "<<delta.count()<<std::endl;

  delta=BuiltinGather256(passes,size,val,mask,ret2);

  std::cout<<"Native Gather256: "<<delta.count()<<std::endl;

  Match(size,ret1,ret2);

  _mm_free(val);
  _mm_free(mask);
  _mm_free(ret1);
  _mm_free(ret2);

  return 0;
}
#pragma once
#include<immintrin.h>

#include"LROCT.h"
#include"Misc.h"


//Based off the paper 
//Fast Bit Gather, Bit Scatter and Bit Permutation Instructions for Commodity Microprocessors
//by Yedidya Hilewitz and Ruby B. Lee 
//DOI:10.1007/s11265-008-0212-8

namespace Butterfly{
  inline __m256i Scatter64x4(__m256i val,__m256i mask){
    using namespace Internal;
    using namespace LROCT;

    __m256i control;

    auto counts=NibblePopCnt(mask);

    counts.bytes=HoriSum64x4(counts.bytes);

    control=LROCT64x32(counts.bytes);
    val=Step64(val,control);
    val=Step32(val,control);

    __m256i upperNibble=_mm256_and_si256(counts.bytes,_mm256_set1_epi8(0x0F));

    control=LROCT16(upperNibble);
    val=Step16(val,control);

    __m256i nibbles=MakeNibble(upperNibble,counts.hiNibble);

    control=LROCT8(nibbles);
    val=StepSub<4>(val,control);

    __m256i bits=MakeBitPair(nibbles,counts.bitPair);

    control=LROCT4(bits);
    val=StepSub<2>(val,control);

    control=MakeSingle(bits,mask);
    val=StepSub<1>(val,control);

    return _mm256_and_si256(val,mask);
  }

  inline __m256i Gather64x4(__m256i val,__m256i mask){
    using namespace Internal;
    using namespace LROCT;

    __m256i control;

    auto counts=NibblePopCnt(mask);

    counts.bytes=HoriSum64x4(counts.bytes);

    __m256i upperNibble=_mm256_and_si256(counts.bytes,_mm256_set1_epi8(0x0F));
    __m256i nibbles=MakeNibble(upperNibble,counts.hiNibble);
    __m256i bits=MakeBitPair(nibbles,counts.bitPair);

    val=_mm256_and_si256(val,mask);

    control=MakeSingle(bits,mask);
    val=StepSub<1>(val,control);

    control=LROCT4(bits);
    val=StepSub<2>(val,control);

    control=LROCT8(nibbles);
    val=StepSub<4>(val,control);

    control=LROCT16(upperNibble);
    val=Step16(val,control);

    control=LROCT64x32(counts.bytes);
    val=Step32(val,control);
    val=Step64(val,control);

    return val;
  }

  inline __m256i Scatter256(__m256i val,__m256i mask){
    using namespace Internal;
    using namespace LROCT;

    __m256i control;

    auto counts=NibblePopCnt(mask);

    counts.bytes=HoriSum256(counts.bytes);

    control=LROCT256x128(counts.bytes);
    val=Step256(val,control);
    val=Step128(val,control);

    control=LROCT64x32(counts.bytes);
    val=Step64(val,control);
    val=Step32(val,control);

    __m256i upperNibble=_mm256_and_si256(counts.bytes,_mm256_set1_epi8(0x0F));

    control=LROCT16(upperNibble);
    val=Step16(val,control);

    __m256i nibbles=MakeNibble(upperNibble,counts.hiNibble);

    control=LROCT8(nibbles);
    val=StepSub<4>(val,control);

    __m256i bits=MakeBitPair(nibbles,counts.bitPair);

    control=LROCT4(bits);
    val=StepSub<2>(val,control);

    control=MakeSingle(bits,mask);
    val=StepSub<1>(val,control);

    return _mm256_and_si256(val,mask);
  }

  inline __m256i Gather256(__m256i val,__m256i mask){
    using namespace Internal;
    using namespace LROCT;

    __m256i control;

    auto counts=NibblePopCnt(mask);

    counts.bytes=HoriSum256(counts.bytes);

    __m256i upperNibble=_mm256_and_si256(counts.bytes,_mm256_set1_epi8(0x0F));
    __m256i nibbles=MakeNibble(upperNibble,counts.hiNibble);
    __m256i bits=MakeBitPair(nibbles,counts.bitPair);

    val=_mm256_and_si256(val,mask);

    control=MakeSingle(bits,mask);
    val=StepSub<1>(val,control);

    control=LROCT4(bits);
    val=StepSub<2>(val,control);

    control=LROCT8(nibbles);
    val=StepSub<4>(val,control);

    control=LROCT16(upperNibble);
    val=Step16(val,control);

    control=LROCT64x32(counts.bytes);
    val=Step32(val,control);
    val=Step64(val,control);

    control=LROCT256x128(counts.bytes);
    val=Step128(val,control);
    val=Step256(val,control);

    return val;
  }
}
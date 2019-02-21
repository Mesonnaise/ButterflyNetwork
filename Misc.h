#pragma once
#include<immintrin.h>

namespace Butterfly::Internal{

  struct popCounts{
    __m256i bytes;
    __m256i hiNibble;
    __m256i bitPair;
  };

  inline popCounts NibblePopCnt(const __m256i &val){
    //Based of an implementation in paper
    //Faster Population Counts Using AVX2 Instructions
    //by Wojciech Muła, Nathan Kurz, Daniel Lemire
    //arXiv:1611.07612

    const __m256i lookupNibble=_mm256_setr_epi64x(
      0x0302020102010100ULL,0x0403030203020201ULL,
      0x0302020102010100ULL,0x0403030203020201ULL);

    const __m256i lookupBit=_mm256_setr_epi64x(
      0x0101010100000000ULL,0x0202020201010101ULL,
      0x0101010100000000ULL,0x0202020201010101ULL);

    const __m256i mask=_mm256_set1_epi8(0x0F);


    __m256i lo=_mm256_and_si256(val,mask);
    __m256i hi=_mm256_and_si256(
      _mm256_srli_epi32(val,4),mask);

    popCounts counts;

    __m256i bitPair=_mm256_slli_epi64(
      _mm256_shuffle_epi8(lookupBit,hi),4);

    counts.bitPair=_mm256_or_si256(
      _mm256_shuffle_epi8(lookupBit,lo),bitPair);

    hi=_mm256_shuffle_epi8(lookupNibble,hi);
    lo=_mm256_shuffle_epi8(lookupNibble,lo);

    counts.hiNibble=hi;
    counts.bytes=_mm256_add_epi8(hi,lo);

    return counts;
  }

  inline __m256i HoriSum256(__m256i val){
    const __m256i hadd=_mm256_set1_epi8(0x01);

    const __m256i shufComp=_mm256_setr_epi64x(
      0xFFFFFFFF0F0B0703ULL,0xFFFFFFFFFFFFFFFFULL,
      0xFFFFFFFF0F0B0703ULL,0xFFFFFFFFFFFFFFFFULL);

    const __m256i shuf2W=_mm256_setr_epi64x(
      0x00000000FFFFFFFFULL,0x0202020201010101ULL,
      0x00000000FFFFFFFFULL,0x0202020201010101ULL);

    const __m256i shuf1H=_mm256_setr_epi64x(
      0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL,
      0x0303030303030303ULL,0x0303030303030303ULL);

    val=_mm256_mullo_epi32(val,hadd);

    __m256i hiByte=_mm256_shuffle_epi8(val,shufComp);

    hiByte=_mm256_mul_epu32(hiByte,hadd);

    val=_mm256_add_epi8(val,_mm256_shuffle_epi8(hiByte,shuf2W));

    hiByte=_mm256_insertf128_si256(hiByte,_mm256_castsi256_si128(hiByte),1);

    val=_mm256_add_epi8(val,_mm256_shuffle_epi8(hiByte,shuf1H));

    return val;
  }

  inline __m256i HoriSum64x4(__m256i val){
    const __m256i hadd=_mm256_set1_epi8(0x01);

    const __m256i shufComp=_mm256_setr_epi64x(
      0x03030303FFFFFFFFULL,0x0B0B0B0BFFFFFFFFULL,
      0x03030303FFFFFFFFULL,0x0B0B0B0BFFFFFFFFULL);

    val=_mm256_mullo_epi32(val,hadd);

    val=_mm256_add_epi8(
      val,_mm256_shuffle_epi8(val,shufComp));

    return val;
  }

  inline __m256i MakeNibble(const __m256i &byte,const __m256i &hiNibble){
    __m256i intern=_mm256_or_si256(byte,_mm256_set1_epi8(0x08));
    __m256i loNibble=_mm256_sub_epi8(intern,hiNibble);
    __m256i nibbles=_mm256_slli_epi64(intern,4);
    return _mm256_or_si256(nibbles,loNibble);
  }

  inline __m256i MakeBitPair(const __m256i &nibbles,const __m256i &bitPair){
    const __m256i maskPair=_mm256_set1_epi8(0x33);

    __m256i lobits=_mm256_sub_epi8(nibbles,bitPair);
    lobits=_mm256_and_si256(lobits,maskPair);

    __m256i bits=_mm256_and_si256(nibbles,maskPair);
    bits=_mm256_slli_epi64(bits,2);
    return _mm256_or_si256(bits,lobits);
  }

  inline __m256i MakeSingle(const __m256i &bits,const __m256i &mask){
    const __m256i bitMask=_mm256_set1_epi8(0x55);

    __m256i single=_mm256_srli_epi64(mask,1);

    single=_mm256_xor_si256(bits,single);
    single=_mm256_xor_si256(single,bitMask);

    return _mm256_and_si256(single,bitMask);
  }


  inline __m256i Step256(__m256i val,__m256i mask){
    __m128i l=_mm256_castsi256_si128(val);
    __m128i h=_mm256_extracti128_si256(val,1);
    __m256i t=_mm256_set_m128i(l,h);

    mask=_mm256_inserti128_si256(mask,_mm256_castsi256_si128(mask),1);

    t=_mm256_xor_si256(val,t);
    t=_mm256_and_si256(t,mask);

    return _mm256_xor_si256(val,t);
  }

  inline __m256i Step128(__m256i val,__m256i mask){
    __m256i t=_mm256_shuffle_epi32(val,0x4E);

    mask=_mm256_permute4x64_epi64(mask,0xAF);

    t=_mm256_xor_si256(val,t);
    t=_mm256_and_si256(t,mask);

    return _mm256_xor_si256(val,t);
  }

  inline __m256i Step64(__m256i val,__m256i mask){
    __m256i t=_mm256_shuffle_epi32(val,0xB1);

    mask=_mm256_shuffle_epi32(mask,0x50);

    t=_mm256_xor_si256(val,t);
    t=_mm256_and_si256(t,mask);

    return _mm256_xor_si256(val,t);
  }

  inline __m256i Step32(__m256i val,__m256i mask){
    const __m256i shuffleVal=_mm256_setr_epi64x(
      0x0504070601000302ULL,0x0D0C0F0E09080B0AULL,
      0x0504070601000302ULL,0x0D0C0F0E09080B0AULL);

    const __m256i shuffleMask=_mm256_setr_epi64x(
      0x0B0A0B0A09080908ULL,0x0F0E0F0E0D0C0D0CULL,
      0x0B0A0B0A09080908ULL,0x0F0E0F0E0D0C0D0CULL);

    __m256i t=_mm256_shuffle_epi8(val,shuffleVal);

    mask=_mm256_shuffle_epi8(mask,shuffleMask);

    t=_mm256_xor_si256(val,t);
    t=_mm256_and_si256(t,mask);

    return _mm256_xor_si256(val,t);
  }

  inline __m256i Step16(__m256i val,__m256i mask){
    const __m256i shuffleVal=_mm256_setr_epi64x(
      0x0607040502030001ULL,0x0E0F0C0D0A0B0809ULL,
      0x0607040502030001ULL,0x0E0F0C0D0A0B0809ULL);

    const __m256i shuffleMask=_mm256_setr_epi64x(
      0x0606040402020000ULL,0x0E0E0C0C0A0A0808ULL,
      0x0606040402020000ULL,0x0E0E0C0C0A0A0808ULL);

    __m256i t=_mm256_shuffle_epi8(val,shuffleVal);

    mask=_mm256_shuffle_epi8(mask,shuffleMask);

    t=_mm256_xor_si256(val,t);
    t=_mm256_and_si256(t,mask);

    return _mm256_xor_si256(val,t);
  }

  template<int shift>
  inline __m256i StepSub(__m256i val,__m256i mask){
    __m256i t=_mm256_srli_epi64(val,shift);
    t=_mm256_and_si256(_mm256_xor_si256(t,val),mask);

    val=_mm256_xor_si256(val,t);
    t=_mm256_slli_epi64(t,shift);
    return _mm256_xor_si256(val,t);
  }
}
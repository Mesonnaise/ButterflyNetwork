#pragma once
#include<immintrin.h>

namespace Butterfly::LROCT{

  inline __m256i LROCT256x128(const __m256i &popCnt){
    const __m256i lookup=_mm256_setr_epi64x(
      0x80C0E0F0F8FCFEFFULL,0x7F3F1F0F07030100ULL,
      0x80C0E0F0F8FCFEFFULL,0x7F3F1F0F07030100ULL);

    const __m256i invFlag=_mm256_setr_epi64x(
      0x8080808080808880ULL,0x8080808080808080ULL,
      0x4040404040404040ULL,0x4040404040404040ULL);

    const __m256i cntMask=_mm256_setr_epi64x(
      0x7F7F7F7F7F7F7F7FULL,0x7F7F7F7F7F7F7F7FULL,
      0x3F3F3F3F3F3F3F3FULL,0x3F3F3F3F3F3F3F3FULL);

    const __m256i cntSub=_mm256_setr_epi64x(
      0x3830282018100800ULL,0x7870686058504840ULL,
      0x3830282018100800ULL,0x3830282018100800ULL);

    const __m256i cntMin=_mm256_setr_epi64x(
      0x0808080808080808ULL,0x0808080808080808ULL,
      0x0808080808080808ULL,0x0808080808080808ULL);

    const __m256i spread=_mm256_setr_epi64x(
      0x0F0F0F0F0F0F0F0FULL,0x0F0F0F0F0F0F0F0FULL,
      0x0707070707070707ULL,0x0F0F0F0F0F0F0F0FULL);

    __m256i intern,inv,mask;

    intern=_mm256_permute4x64_epi64(popCnt,0x24);

    intern=_mm256_shuffle_epi8(intern,spread);

    inv=_mm256_and_si256(intern,invFlag);
    inv=_mm256_cmpeq_epi8(inv,invFlag);

    intern=_mm256_and_si256(intern,cntMask);
    intern=_mm256_subs_epu8(intern,cntSub);
    intern=_mm256_min_epu8(intern,cntMin);

    mask=_mm256_shuffle_epi8(lookup,intern);
    mask=_mm256_xor_si256(mask,inv);
    return mask;
  }

  inline __m256i LROCT64x32(const __m256i &popCnt){
    const __m256i lookup=_mm256_setr_epi64x(
      0x80C0E0F0F8FCFEFFULL,0x7F3F1F0F07030100ULL,
      0x80C0E0F0F8FCFEFFULL,0x7F3F1F0F07030100ULL);

    const __m256i invFlag=_mm256_setr_epi64x(
      0x2020202020202020ULL,0x1010101010101010ULL,
      0x2020202020202020ULL,0x1010101010101010ULL);

    const __m256i cntMask=_mm256_setr_epi64x(
      0x1F1F1F1F1F1F1F1FULL,0x0F0F0F0F0F0F0F0FULL,
      0x1F1F1F1F1F1F1F1FULL,0x0F0F0F0F0F0F0F0FULL);

    const __m256i cntSub=_mm256_setr_epi64x(
      0x1810080018100800ULL,0x0800080008000800ULL,
      0x1810080018100800ULL,0x0800080008000800ULL);

    const __m256i cntMin=_mm256_setr_epi64x(
      0x0808080808080808ULL,0x0808080808080808ULL,
      0x0808080808080808ULL,0x0808080808080808ULL);

    const __m256i spread=_mm256_setr_epi64x(
      0x0B0B0B0B03030303ULL,0x0D0D090905050101ULL,
      0x0B0B0B0B03030303ULL,0x0D0D090905050101ULL);

    __m256i intern,inv,mask;

    intern=_mm256_shuffle_epi8(popCnt,spread);

    inv=_mm256_and_si256(intern,invFlag);
    inv=_mm256_cmpeq_epi8(inv,invFlag);

    intern=_mm256_and_si256(intern,cntMask);
    intern=_mm256_subs_epu8(intern,cntSub);
    intern=_mm256_min_epu8(intern,cntMin);

    mask=_mm256_shuffle_epi8(lookup,intern);
    mask=_mm256_xor_si256(mask,inv);

    return mask;
  }

  inline __m256i LROCT16(const __m256i &popCnt){
    const __m256i lookup=_mm256_setr_epi64x(
      0x80C0E0F0F8FCFEFFULL,0x7F3F1F0F07030100ULL,
      0x80C0E0F0F8FCFEFFULL,0x7F3F1F0F07030100ULL);

    return _mm256_shuffle_epi8(lookup,popCnt);
  }

  inline __m256i LROCT8(const __m256i &popCnt){
    const __m256i mask=_mm256_set1_epi8(0x7);

    const __m256i lookup=_mm256_setr_epi64x(
      0x07030100080C0E0FULL,0x07030100080C0E0FULL,
      0x07030100080C0E0FULL,0x07030100080C0E0FULL);

    __m256i intern=_mm256_and_si256(popCnt,mask);
    return _mm256_shuffle_epi8(lookup,intern);
  }

  inline __m256i LROCT4(const __m256i &popCnt){
    const __m256i mask=_mm256_set1_epi8(0x33);

    __m256i intern=_mm256_and_si256(popCnt,mask);
    intern=_mm256_xor_si256(intern,_mm256_srli_epi64(intern,1));
    return _mm256_andnot_si256(intern,mask);
  }
}
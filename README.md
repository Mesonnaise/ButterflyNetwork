# ButterflyNetwork

The AVX2 implementation are designed to be replacements for the BMI2 instructions pdep and pext when used in memory scans. The functions come in two variants 64x4, and 256. The 64bits by 4 are used as a direct substitute for pdep and pext. The 256bit variant is better suited for compaction/expansion of bit arrays and other succinct structures. 

The implementation comes with a Visual Studio solution for validation testing. 

## Reference

Fast Bit Gather, Bit Scatter and Bit Permutation Instructions for Commodity Microprocessors  
by Yedidya Hilewitz and Ruby B. Lee  
DOI:10.1007/s11265-008-0212-8

Faster Population Counts Using AVX2 Instructions  
by Wojciech Muła, Nathan Kurz, Daniel Lemire  
arXiv:1611.07612

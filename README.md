# ci-neuralnetwork
A Neural Network implementation in jupyter for TUM CI class.

The implementation is aiming to not use any sklearn/caffee/whatever functions to manipulate data or lean.

## Prerequesites
A jupyter installation for Python 3 is recommended.

Also make sure to have the following libs installed:
- numpy 
- matplotlib

## Port to C
The Port to C will reflect exaclty the working jupyter implementation. 
*It does, however, not yet do so*!

The code is compilable with MinGW GCC and the following call:
`gcc -g -Wall -O3 -DNDEBUG -static -std=c99 -pipe main.c -lm`

*Have Fun* 
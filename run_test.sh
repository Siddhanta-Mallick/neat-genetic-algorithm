#!/bin/bash

g++ tests/test_cycle.cpp \
    -std=c++20 \
    -Iexternal \
    -Isrc \
    -Iinclude \
    -o test

./test

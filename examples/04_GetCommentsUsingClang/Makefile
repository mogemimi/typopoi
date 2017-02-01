#!/bin/make

# e.g. export CLANG_DIR="/path/to/clang+llvm-3.8.0-x86_64-apple-darwin"
CLANG_DIR = /usr/local/opt/clang+llvm

CXX = clang++
CXXFLAGS = \
	-std=c++14 \
	-O2 \
	-Wall \
	-I../.. \
	-I$(CLANG_DIR)/include \
	-D__STDC_CONSTANT_MACROS \
	-D__STDC_LIMIT_MACROS \
	-fno-exceptions \
	-fno-rtti
LDFLAGS = -L$(CLANG_DIR)/lib
PRODUCT_NAME = 04_GetCommentsUsingClang
SOURCES = \
	../../SpellChecker.cpp \
	main.cpp
LDLIBS = \
	-lLLVMCore \
	-lLLVMOption \
	-lLLVMSupport \
	-lLLVMMCParser \
	-lLLVMBitReader \
	-lLLVMMC \
	-lclangAnalysis \
	-lclangAST \
	-lclangASTMatchers \
	-lclangBasic \
	-lclangDriver \
	-lclangEdit \
	-lclangFrontend \
	-lclangParse \
	-lclangLex \
	-lclangRewrite \
	-lclangRewriteFrontend \
	-lclangSema \
	-lclangSerialization \
	-lclangTooling \
	-lz \
	-lpthread \
	-ledit \
	-lcurses \
	-lm

build:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PRODUCT_NAME) $(SOURCES) $(LDLIBS)

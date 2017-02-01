# Example 05. Get variables from AST

## Download LLVM Clang

- http://releases.llvm.org/download.html

#### Mac OS X

```shell
cd path/to
wget http://releases.llvm.org/3.9.0/clang+llvm-3.9.0-x86_64-apple-darwin.tar.xz
tar xaf clang+llvm-3.9.0-x86_64-apple-darwin
```

## Build

```shell
cd typopoi/examples/05_GetVariablesFromAST
make build CLANG_DIR=/path/to/clang+llvm-3.9.0
```

#### Mac OS X

```shell
cd typopoi/examples/05_GetVariablesFromAST
make build CLANG_DIR=/path/to/clang+llvm-3.9.0-x86_64-apple-darwin
```

## Run

#### Get variables and comments from source code

```shell
.05_GetVariablesFromAST main.cpp --
```

Result:

```shell
MyToolCategory
CommonHelp
MoreHelp
// unnamed namespace

main
```

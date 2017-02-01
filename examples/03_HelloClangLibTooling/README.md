# Example 03. Hello Clang LibTooling

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
cd typopoi/examples/03_HelloClangLibTooling
make build CLANG_DIR=/path/to/clang+llvm-3.9.0
```

#### Mac OS X

```shell
cd typopoi/examples/03_HelloClangLibTooling
make build CLANG_DIR=/path/to/clang+llvm-3.9.0-x86_64-apple-darwin
```

## Run

```shell
./03_HelloClangLibTooling -help
```

Result:

```
$ ./03_HelloClangLibTooling -help
USAGE: 03_HelloClangLibTooling [options] <source0> [... <sourceN>]

OPTIONS:

Generic Options:

  -help                      - Display available options (-help-hidden for more)
  -help-list                 - Display list of available options (-help-list-hidden for more)
  -version                   - Display the version of this program

my-tool options:

  -extra-arg=<string>        - Additional argument to append to the compiler command line
  -extra-arg-before=<string> - Additional argument to prepend to the compiler command line
  -p=<string>                - Build path
```

# Example 04. Get Comments Using LibTooling

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
cd typopoi/examples/04_GetCommentsUsingClang
make build CLANG_DIR=/path/to/clang+llvm-3.9.0
```

#### Mac OS X

```shell
cd typopoi/examples/04_GetCommentsUsingClang
make build CLANG_DIR=/path/to/clang+llvm-3.9.0-x86_64-apple-darwin
```

## Run

#### Get comments from a source code

```shell
./04_GetCommentsUsingClang main.cpp ../../SpellChecker.cpp ../../SpellChecker.h --
```

Result:

```shell
/path/to/typopoi/examples/04_GetCommentsUsingClang/main.cpp:
    // unnamed namespace

/path/to/typopoi/examples/04_GetCommentsUsingClang/../../SpellChecker.cpp:
    // Copyright (c) 2015-2017 mogemimi. Distributed under the MIT license.
...
    // namespace typopoi

/path/to/typopoi/examples/04_GetCommentsUsingClang/../../SpellChecker.h:
    // Copyright (c) 2015-2017 mogemimi. Distributed under the MIT license.
    ///@brief `true` if the word is correctly spelled; `false` otherwise.
    ///@brief Add a word to dictionary.
    ///@param word UTF-8 encoded string
    ///@brief Add a word to dictionary.
    ///@param word UTF-8 encoded string
    ///@brief Remove a word from dictionary.
    ///@param word UTF-8 encoded string
    ///@brief Convert a UTF-8 encoded string into a UTF-32 encoded string.
    ///@brief Convert a UTF-32 encoded string into a UTF-8 encoded string.
    ///@brief islower() function for UTF-32 encoding.
    ///@brief isupper() function for UTF-32 encoding.
    ///@brief tolower() function for UTF-32 encoding.
    ///@brief toupper() function for UTF-32 encoding.
    // namespace typopoi
```

#### Show help messages

```shell
./04_GetCommentsUsingClang -help
```

Result:

```
$ ./04_GetCommentsUsingClang -help
USAGE: 04_GetCommentsUsingClang [options] <source0> [... <sourceN>]

OPTIONS:

Generic Options:

  -help                      - Display available options (-help-hidden for more)
  -help-list                 - Display list of available options (-help-list-hidden for more)
  -version                   - Display the version of this program

...

More help text...%
```

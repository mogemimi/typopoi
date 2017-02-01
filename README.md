# typopoi

**TypoPoi** is a spell checker written in C++14.
It provides a library and a standalone tool to easily find spelling mistakes in a text/source code.
The library was designed for use with static code analysis tools.

TypoPoi is also open source and distributed under the [MIT License](https://opensource.org/licenses/MIT).
Feel free to fork, submit a pull request or modify anywhere you like!

## Spell checking library

The spell checking library is a fast, easy-to-use, has no dependencies.
It is constituted of the following files:

- `SpellChecker.h`
- `SpellChecker.cpp`

#### Usage

```cpp
#include "SpellChecker.h"
```

```cpp
typopoi::SpellChecker spellChecker;
spellChecker.AddWord(u8"defer");
spellChecker.AddWord(u8"deferred");
spellChecker.AddWord(u8"refer");
spellChecker.AddWord(u8"referred");
spellChecker.AddWord(u8"reference");

auto word = u8"defered";
auto result = spellChecker.Suggest(word);

if (result.correctlySpelled) {
    std::cout << "The word '" << word << "' is correctly spelled" << std::endl;
}

std::cout << "Did you mean:" << std::endl;
for (auto & suggestion : result.suggestions) {
    std::cout << suggestion << std::endl;
}
```

To detect spelling errors in your code, you can specify the identifiers (which denote functions, variables and types etc.) directly:

```cpp
// Did you mean: "GetDeferredRenderer" or "GetDeferRenderer"
spellChecker.Suggest(u8"GetDeferedRenderer");

// Did you mean: "reference_count"
spellChecker.Suggest(u8"refrence_count");
```

#### API

## Build for Mac OS X

```shell
# Build executable with Xcode
xcodebuild -project projects/typopoi.xcodeproj -target libtypopoi -configuration Release
```

```shell
# Build executable with Xcode
xcodebuild -project projects/typopoi.xcodeproj -target typopoi -configuration Release

# Installation
cp build/Release/typopoi typopoi

# Run
.typopoi -help
```

## Standalone command line tool

`tools` directory

## Thanks

The following libraries and/or open source projects were used in typopoi:

* [LLVM Clang](http://clang.llvm.org/)
* [Project Gutenberg](https://www.gutenberg.org/)

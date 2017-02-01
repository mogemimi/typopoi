// Copyright (c) 2015 mogemimi. Distributed under the MIT license.

#pragma once

#include "SpellChecker.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace typopoi {

//struct SourceLocation {
//    std::string filePath;
//};

struct Typo {
    //SourceLocation location;
    //SourceRange range;
    std::string misspelledWord;
    std::vector<std::string> corrections;
};

class TypoCache final {
public:
    TypoCache();

    bool exists(const std::string& typo) const;

    void setCapacity(size_t capacity);

    void insert(Typo && typo);

private:
    struct TypoCacheInfo {
        std::string misspelledWord;
        int count;
    };

    std::vector<TypoCacheInfo> typos;
    std::size_t maxCacheSize;
};

class TypoMan final {
private:
    std::shared_ptr<typopoi::SpellChecker> spellChecker;
    TypoCache cache;
    std::function<void(const Typo&)> onFoundTypo;
    int minimumWordSize;
    int maxCorrectWordCount;
    bool isStrictWhiteSpace;
    bool isStrictHyphen;
    bool isStrictLetterCase;
    bool isCacheEnabled;

public:
    TypoMan() noexcept;

    void computeFromWord(const std::string& word);

    void setSpellChecker(const std::shared_ptr<SpellChecker>& spellChecker);

    void setMinimumWordSize(int wordSize);

    void setMaxCorrectWordCount(int maxCount);

    void setCacheEnabled(bool cacheEnabled);

    void setCacheSize(std::size_t cacheSize);

    void setFoundCallback(std::function<void(const Typo&)> callback);
};

} // namespace typopoi

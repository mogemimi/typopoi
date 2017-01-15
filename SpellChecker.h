// Copyright (c) 2015-2017 mogemimi. Distributed under the MIT license.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

namespace typopoi {

struct SpellCheckResult {
    std::vector<std::string> suggestions;

    ///@brief `true` if the word is correctly spelled; `false` otherwise.
    bool correctlySpelled;
};

class SpellChecker {
public:
    ///@brief Add a word to dictionary.
    ///@param word UTF-8 encoded string
    SpellCheckResult Suggest(const std::string& word);

    ///@brief Add a word to dictionary.
    ///@param word UTF-8 encoded string
    void AddWord(const std::string& word);

    ///@brief Remove a word from dictionary.
    ///@param word UTF-8 encoded string
    void RemoveWord(const std::string& word);

private:
    std::unordered_map<uint32_t, std::vector<std::u32string>> hashedDictionary;
};

///@brief Convert a UTF-8 encoded string into a UTF-32 encoded string.
std::u32string ToUtf32(const std::string& utf8);

///@brief Convert a UTF-32 encoded string into a UTF-8 encoded string.
std::string ToUtf8(const std::u32string& utf32);

///@brief islower() function for UTF-32 encoding.
bool IsLowerUtf32(char32_t c) noexcept;

///@brief isupper() function for UTF-32 encoding.
bool IsUpperUtf32(char32_t c) noexcept;

///@brief tolower() function for UTF-32 encoding.
char32_t ToLowerUtf32(char32_t c) noexcept;

///@brief toupper() function for UTF-32 encoding.
char32_t ToUpperUtf32(char32_t c) noexcept;

} // namespace typopoi

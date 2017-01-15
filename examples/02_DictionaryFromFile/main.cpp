#include "SpellChecker.h"
#include <iostream>
#include <fstream>

using typopoi::SpellChecker;

bool ReadDictionaryFile(
    const std::string& path,
    const std::function<void(const std::string&)>& callback)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "error: Cannot open the file. " << path << std::endl;
        return false;
    }

    std::istreambuf_iterator<char> start(input);
    std::istreambuf_iterator<char> end;

    std::string word;
    for (; start != end; ++start) {
        auto c = *start;
        if (c == '\r' || c == '\n' || c == '\0') {
            if (!word.empty()) {
                callback(word);
            }
            word.clear();
            continue;
        }
        word += c;
    }
    if (!word.empty()) {
        // TODO: The word must be a UTF-8 encoded string but this function
        // does not validate that the word is UTF-8 string.
        callback(word);
    }
    return true;
}

int main()
{
    // NOTE: Please specify your dictionary file
    std::string filePath = "MyDictionary.txt";

    SpellChecker spellChecker;

    // NOTE: Construct a dictionary from the file
    auto addWord = [&](const std::string& word) {
        spellChecker.AddWord(word);
    };
    if (!ReadDictionaryFile(filePath, addWord)) {
        return 1;
    }

    // NOTE: Check spelling error
    auto word = u8"Defered";
    auto result = spellChecker.Suggest(word);

    // NOTE: Show search result
    if (result.correctlySpelled) {
        std::cout << "'" << word << "' is found. (exact match)" << std::endl;
    }
    else if (result.suggestions.empty()) {
        std::cout << "'" << word << "' is not found." << std::endl;
    }
    else {
        std::cout << "'" << word << "' " << "Did you mean..." << std::endl;
        for (auto & suggestion : result.suggestions) {
            std::cout << "  " << suggestion << std::endl;
        }
    }

    return 0;
}

#include "SpellChecker.h"
#include <iostream>

int main()
{
    typopoi::SpellChecker spellChecker;
    spellChecker.AddWord(u8"hello");
    spellChecker.AddWord(u8"world");

    auto suggest = [&](const std::string& word) -> std::string {
        auto result = spellChecker.Suggest(word);
        if (result.correctlySpelled) {
            // NOTE: 'word' is correct.
            return word;
        }
        if (result.suggestions.empty()) {
            // NOTE: 'word' is not found in the directory.
            return word;
        }
        // NOTE: 'word' is probably misspelled word.
        return result.suggestions.front();
    };

    // NOTE: "Hello, world!"
    std::cout << suggest(u8"Hell, would!") << std::endl;

    return 0;
}

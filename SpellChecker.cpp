// Copyright (c) 2015-2017 mogemimi. Distributed under the MIT license.

#include "SpellChecker.h"
#include <algorithm>
#include <cassert>
#include <codecvt>
#include <cstdint>
#include <locale>
#include <string>
#include <unordered_map>
#include <utility>

namespace typopoi {
namespace {

struct nullopt_t final {
    struct init { constexpr init() = default; };
    constexpr explicit nullopt_t(init) noexcept {}
};
constexpr nullopt_t nullopt{ nullopt_t::init{} };

template <typename T>
struct optional {
private:
    T data;
    bool valid = false;

public:
    constexpr optional() : data(), valid(false) {}
    constexpr optional(const nullopt_t&) : data(), valid(false) {}
    optional(const optional&) = default;
    optional(optional &&) = default;
    constexpr optional(const T& v) : data(v), valid(true) {}
    constexpr optional(T && v) : data(std::move(v)), valid(true) {}

    optional & operator=(const nullopt_t&)
    {
        valid = false;
        data.~T();
        return *this;
    }

    optional & operator=(const optional&) = default;
    optional & operator=(optional &&) = default;

    optional & operator=(const T& v)
    {
        this->valid = true;
        this->data = v;
        return *this;
    }

    optional & operator=(T && v)
    {
        this->valid = true;
        this->data = std::move(v);
        return *this;
    }

    T const* operator->() const noexcept
    {
        assert(valid);
        return &data;
    }

    T* operator->() noexcept
    {
        assert(valid);
        return &data;
    }

    const T& operator*() const
    {
        assert(valid);
        return data;
    }

    T & operator*()
    {
        assert(valid);
        return data;
    }

    explicit operator bool() const noexcept
    {
        return valid;
    }

    T const& value() const
    {
        assert(valid);
        return data;
    }

    T & value()
    {
        assert(valid);
        return data;
    }
};

struct SignatureHashAlphabet {
    static constexpr uint32_t MaxHashLength = 28;

    static uint32_t Hash(const std::u32string& word);
};

constexpr uint32_t SignatureHashAlphabet::MaxHashLength;

uint32_t SignatureHashAlphabet::Hash(const std::u32string& word)
{
    // NOTE:
    // This algorithm is based on Boitsov's Signature Hashing Algorithm in
    // Boitsov, L.M. "Using Signature Hashing for Approximate String Matching",
    // Computational Mathematics and Modeling (2002) 13: 314.

    // NOTE:
    // '#' is number 0 to 9.
    // '@' is symbols and other characters.
    //
    //   ABCDEFGHIJKLMNOPQRSTUVWXYZ#@
    //   abcdefghijklmnopqrstuvwxyz#@
    // 0b0000000000000000000000000000
    //
    // (Example) Hash of "Blade Runner":
    //
    //   ABCDEFGHIJKLMNOPQRSTUVWXYZ#@
    //   abcdefghijklmnopqrstuvwxyz#@
    // 0b1101100000010100010010000000
    constexpr int offsets[128] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 27, 26, 25,
        24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 27, 26, 25, 24, 23,
        22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 0, 0, 0, 0, 0
    };

    uint32_t hash = 0;
    for (auto & c : word) {
        if (c < 128) {
            auto offset = offsets[c];
            hash |= (1 << offset);
        }
        else {
            hash |= (1 << (c % MaxHashLength));
        }
    }
    return hash;
}

double MatchCharacter(char32_t misspell, char32_t suggestion)
{
    constexpr auto FullMatchScore = 1.0;
    constexpr auto CapitalMatchScore = 0.9;
    constexpr auto NoMatch = 0.0;

    // NOTE:
    //
    // matchCharacter(a, a) == 1.0
    // matchCharacter(a, A) == 0.9
    // matchCharacter(A, a) == 1.0
    // matchCharacter(A, A) == 1.0
    //
    // matchCharacter(b, a) == 0.0
    // matchCharacter(b, A) == 0.0
    // matchCharacter(B, a) == 0.0
    // matchCharacter(B, A) == 0.0

    if (misspell == suggestion) {
        return FullMatchScore;
    }
    if (IsUpperUtf32(suggestion)) {
        if (ToUpperUtf32(misspell) == ToUpperUtf32(suggestion)) {
            return CapitalMatchScore;
        }
    }
    else {
        if (ToLowerUtf32(misspell) == suggestion) {
            return FullMatchScore;
        }
    }
    return NoMatch;
}

double ComputeLCSLengthFuzzy(
    const std::u32string& text1,
    const std::u32string& text2,
    int distanceThreshold)
{
    // NOTE:
    // This algorithm is based on Myers's An O((M+N)D) Greedy Algorithm in
    // Myers, E.W. "An O(ND)Difference Algorithm and Its Variations",
    // Algorithmica (1986), pages 251-266.

    if (text1.empty() || text2.empty()) {
        return 0.0;
    }

    assert(distanceThreshold > 0);

    const auto M = static_cast<int>(text1.size());
    const auto N = static_cast<int>(text2.size());

    const auto maxD = M + N;
    const auto offset = N;

    // NOTE:
    // There is no need to initialize with the zero value for array elements,
    // but you have to assign the zero value to `vertices[1 + offset]`.
    std::vector<int> vertices(M + N + 1);
    vertices[1 + offset] = 0;

    std::vector<double> lcsLengths(M + N + 1);
    lcsLengths[1 + offset] = 0;

    for (int d = 0; d <= maxD; ++d) {
        if (d > distanceThreshold) {
            return 0.0;
        }

        const int startK = -std::min(d, (N * 2) - d);
        const int endK = std::min(d, (M * 2) - d);

        assert((-N <= startK) && (endK <= M));
        assert(std::abs(startK % 2) == (d % 2));
        assert(std::abs(endK % 2) == (d % 2));
        assert((d > N) ? (startK == -(N * 2 - d)) : (startK == -d));
        assert((d > M) ? (endK == (M * 2 - d)) : (endK == d));

        for (int k = startK; k <= endK; k += 2) {
            assert((-N <= k) && (k <= M));
            assert(std::abs(k % 2) == (d % 2));

            const auto kOffset = k + offset;

            int x = 0;
            double lcsLength = 0.0;
            if (k == startK) {
                // NOTE: Move directly from vertex(x, y - 1) to vertex(x, y)
                x = vertices[kOffset + 1];
                lcsLength = lcsLengths[kOffset + 1];
            }
            else if (k == endK) {
                // NOTE: Move directly from vertex(x - 1, y) to vertex(x, y)
                x = vertices[kOffset - 1] + 1;
                lcsLength = lcsLengths[kOffset - 1];
            }
            else if (vertices[kOffset - 1] < vertices[kOffset + 1]) {
                // NOTE: Move from vertex(k + 1) to vertex(k)
                // vertex(k + 1) is ahead of vertex(k - 1).
                assert(-N < k && k < M);
                assert((k != -d) && (k != -N));
                assert((k != d) && (k != M));
                x = vertices[kOffset + 1];
                lcsLength = lcsLengths[kOffset + 1];
            }
            else {
                // NOTE: Move from vertex(k - 1) to vertex(k)
                // vertex(k - 1) is ahead of vertex(k + 1).
                assert(-N < k && k < M);
                assert((k != -d) && (k != -N));
                assert((k != d) && (k != M));
                assert(vertices[kOffset - 1] >= vertices[kOffset + 1]);
                x = vertices[kOffset - 1] + 1;
                lcsLength = lcsLengths[kOffset - 1];
            }

            // NOTE: `k` is defined from `x - y = k`.
            int y = x - k;
            assert(x >= 0 && y >= 0);

            while (x < M && y < N) {
                auto score = MatchCharacter(text1[x], text2[y]);
                assert((0.0 <= score) && (score <= 1.0));
                if (score == 0.0) {
                    break;
                }
                // NOTE: This loop finds a possibly empty sequence
                // of diagonal edges called a 'snake'.
                x += 1;
                y += 1;
                lcsLength += score;
            }

            if (x >= M && y >= N) {
                return lcsLength;
            }

            vertices[kOffset] = x;
            lcsLengths[kOffset] = lcsLength;
        }
    }
    return 0.0;
}

double ClosestMatchSimilarity(
    const std::u32string& text1,
    const std::u32string& text2,
    int distanceThreshold)
{
    if (text1.empty() && text2.empty()) {
        return 1.0;
    }
    auto lcs = ComputeLCSLengthFuzzy(text1, text2, distanceThreshold);
    auto maxLength = static_cast<double>(std::max(text1.size(), text2.size()));
    assert(maxLength >= 1.0);
    assert(maxLength != 0);
    return lcs / maxLength;
}

double ClosestMatchSimilarity(
    const std::u32string& text1,
    const std::u32string& text2)
{
    return ClosestMatchSimilarity(text1, text2, static_cast<int>(text1.size() + text2.size()));
}

std::size_t ComputeGapSize(std::size_t a, std::size_t b)
{
    if (a > b) {
        assert((a - b) <= a);
        return a - b;
    }
    assert((b - a) <= b);
    return b - a;
}

struct SpellSuggestion {
    std::u32string word;
    typopoi::optional<double> similarity;
};

struct SpellCheckResultInternal {
    std::vector<SpellSuggestion> suggestions;

    ///@brief `true` if the word is correctly spelled; `false` otherwise.
    bool correctlySpelled;
};

void SpellCheckInternal(
    const std::u32string& input,
    const std::vector<std::u32string>& dictionary,
    std::vector<SpellSuggestion> & suggestions,
    bool & exactMatching,
    std::size_t inputWordSize,
    std::size_t & gapSizeThreshold,
    double & similarityThreshold,
    int distanceThreshold)
{
    for (auto & word : dictionary) {
        const auto gapSize = ComputeGapSize(word.size(), inputWordSize);
        if (gapSize > gapSizeThreshold) {
            continue;
        }

        const auto similarity = ClosestMatchSimilarity(input, word, distanceThreshold);
        if (similarity == 1.0) {
            // NOTE: exact matching
            SpellSuggestion suggestion;
            suggestion.word = word;
            suggestion.similarity = similarity;
            suggestions.insert(std::begin(suggestions), std::move(suggestion));
            exactMatching = true;
            break;
        }
        else if (similarity >= similarityThreshold) {
            similarityThreshold = std::max(similarity, similarityThreshold);
            gapSizeThreshold = std::max<std::size_t>(gapSize, 1);
            assert(similarityThreshold <= 1.0);
            SpellSuggestion suggestion;
            suggestion.word = word;
            suggestion.similarity = similarity;
            suggestions.push_back(std::move(suggestion));
        }
    }
}

std::size_t ComputePrefixLength(const std::u32string& a, const std::u32string& b)
{
    const auto minSize = std::min(a.size(), b.size());
    std::size_t prefixLength = 0;
    for (std::size_t i = 0; i < minSize; ++i) {
        assert(i < a.size());
        assert(i < b.size());
        if (a[i] != b[i]) {
            break;
        }
        ++prefixLength;
    }
    return prefixLength;
}

std::size_t ComputeSuffixLength(const std::u32string& a, const std::u32string& b)
{
    const auto minSize = std::min(a.size(), b.size());
    std::size_t suffixLength = 0;
    for (std::size_t i = 1; i <= minSize; ++i) {
        assert(i <= a.size());
        assert(i <= b.size());
        if (a[a.size() - i] != b[b.size() - i]) {
            break;
        }
        ++suffixLength;
    }
    return suffixLength;
}

int LevenshteinDistance_ReplacementCost1(
    const std::u32string& text1,
    const std::u32string& text2)
{
    // NOTE:
    // This algorithm is based on dynamic programming, using only linear space.
    // It is O(N^2) time and O(N) space algorithm.

    const auto rows = static_cast<int>(text1.size()) + 1;
    const auto columns = static_cast<int>(text2.size()) + 1;
    std::vector<int> c1(columns);
    std::vector<int> c2(columns);

    for (int i = 0; i < columns; ++i) {
        c1[i] = i;
    }

    for (int row = 1; row < rows; row++) {
        c2[0] = row;
        for (int column = 1; column < columns; column++) {
            if (text1[row - 1] == text2[column - 1]) {
                c2[column] = c1[column - 1];
            }
            else {
                // NOTE: The cost of 'insertion', 'deletion' and 'substitution' operations is 1, not 2.
                c2[column] = std::min(c1[column - 1], std::min(c1[column], c2[column - 1])) + 1;
            }
        }
        // NOTE: Use faster swap() function instead of "c1 = c2;" to faster
        std::swap(c1, c2);
    }
    return c1.back();
}

void SortSuggestions(const std::u32string& word, std::vector<SpellSuggestion>& suggestions)
{
    for (auto & suggestion : suggestions) {
        if (!suggestion.similarity) {
            suggestion.similarity = ClosestMatchSimilarity(word, suggestion.word);
        }
    }

    std::sort(std::begin(suggestions), std::end(suggestions), [&](const SpellSuggestion& a, const SpellSuggestion& b) {
        assert(a.similarity);
        assert(b.similarity);
        if (*a.similarity != *b.similarity) {
            return *a.similarity > *b.similarity;
        }
        const auto distanceA = LevenshteinDistance_ReplacementCost1(word, a.word);
        const auto distanceB = LevenshteinDistance_ReplacementCost1(word, b.word);
        if (distanceA != distanceB) {
            return distanceA < distanceB;
        }
        const auto prefixA = ComputePrefixLength(word, a.word);
        const auto prefixB = ComputePrefixLength(word, b.word);
        if (prefixA != prefixB) {
            return prefixA > prefixB;
        }
        const auto gapA = ComputeGapSize(word.size(), a.word.size());
        const auto gapB = ComputeGapSize(word.size(), b.word.size());
        if (gapA != gapB) {
            return gapA < gapB;
        }
        const auto suffixA = ComputeSuffixLength(word, a.word);
        const auto suffixB = ComputeSuffixLength(word, b.word);
        return suffixA > suffixB;
    });
}

std::vector<uint32_t> GenerateBitmasks(uint32_t hash, int maxHashLength)
{
    std::vector<int> positiveBits;
    std::vector<int> negativeBits;

    for (int i = 1; i <= maxHashLength; ++i) {
        if (((hash >> i) & 0b1) == 0b1) {
            positiveBits.push_back(i);
        }
        else {
            negativeBits.push_back(i);
        }
    }

    std::vector<uint32_t> bitmasks;
    bitmasks.reserve(maxHashLength + (positiveBits.size() * negativeBits.size()));

    for (int i = 0; i <= maxHashLength; ++i) {
        uint32_t bitmask = ((static_cast<uint32_t>(1) << i) >> 1);
        bitmasks.push_back(bitmask);
    }

    for (auto positive : positiveBits) {
        uint32_t bitmaskPositive = (static_cast<uint32_t>(1) << positive);
        for (auto negative : negativeBits) {
            uint32_t bitmaskNegative = (static_cast<uint32_t>(1) << negative);
            uint32_t bitmask = bitmaskPositive | bitmaskNegative;
            assert(bitmask != 0);
            bitmasks.push_back(bitmask);
        }
    }
    return bitmasks;
}

template <class SignatureHashing>
SpellCheckResultInternal SpellCheckBySignatureHashing(
    const std::u32string& input,
    const std::unordered_map<uint32_t, std::vector<std::u32string>>& hashedDictionary,
    const int distanceThreshold)
{
    std::size_t gapSizeThreshold = 2;

    SpellCheckResultInternal result;
    result.correctlySpelled = false;

    const auto inputSignatureHash = SignatureHashing::Hash(input);
    const auto inputWordSize = input.size();

    const auto sizeAsDouble = static_cast<double>(inputWordSize);
    double similarityThreshold = std::max((sizeAsDouble - std::min(sizeAsDouble, 2.0)) / sizeAsDouble, 0.5);

    const auto bitmasks = GenerateBitmasks(inputSignatureHash, SignatureHashing::MaxHashLength);

    for (auto bitmask : bitmasks) {
        const uint32_t xorBits = inputSignatureHash ^ bitmask;
        auto iter = hashedDictionary.find(xorBits);
        if (iter == std::end(hashedDictionary)) {
            continue;
        }
        const auto& dictionary = iter->second;

        bool exactMatching = false;

        SpellCheckInternal(
            input,
            dictionary,
            result.suggestions,
            exactMatching,
            inputWordSize,
            gapSizeThreshold,
            similarityThreshold,
            distanceThreshold);

        if (exactMatching) {
            result.correctlySpelled = true;
            break;
        }
    }

    return result;
}

template <class SignatureHashing>
typopoi::optional<SpellSuggestion> ExistWordBySignatureHashing(
    const std::u32string& input,
    const std::unordered_map<uint32_t, std::vector<std::u32string>>& hashedDictionary)
{
    std::size_t gapSizeThreshold = 1;
    const int distanceThreshold = 1;

    SpellCheckResultInternal result;
    result.correctlySpelled = false;

    const auto inputWordSize = input.size();

    double similarityThreshold = 0.8;

    const auto inputSignatureHash = SignatureHashing::Hash(input);

    auto iter = hashedDictionary.find(inputSignatureHash);
    if (iter == std::end(hashedDictionary)) {
        return typopoi::nullopt;
    }
    const auto& dictionary = iter->second;

    typopoi::optional<SpellSuggestion> currentSuggestion;

    for (auto & word : dictionary) {
        const auto gapSize = ComputeGapSize(word.size(), inputWordSize);
        if (gapSize > gapSizeThreshold) {
            continue;
        }

        const auto similarity = ClosestMatchSimilarity(input, word, distanceThreshold);
        if (similarity == 1.0) {
            // NOTE: exact matching
            SpellSuggestion suggestion;
            suggestion.word = word;
            suggestion.similarity = similarity;
            return suggestion;
        }
        else if (similarity >= similarityThreshold) {
            similarityThreshold = std::max(similarity, similarityThreshold);
            gapSizeThreshold = std::max<std::size_t>(gapSize, 1);
            assert(similarityThreshold <= 1.0);
            SpellSuggestion suggestion;
            suggestion.word = word;
            suggestion.similarity = similarity;
            currentSuggestion = std::move(suggestion);
        }
    }
    return currentSuggestion;
}

enum class LetterCase {
    Lowercase,
    Uppercase,
    Titlecase,
};

LetterCase GetLetterCase(const std::u32string& word)
{
    bool isLowerCase = true;
    bool isUpperCase = true;
    bool isUpperCamelCase = true;
    if (word.empty() || (IsUpperUtf32(word.front()) == 0)) {
        isUpperCamelCase = false;
    }
    for (auto c : word) {
        if (IsLowerUtf32(c) != 0) {
            isUpperCase = false;
        }
        if (IsUpperUtf32(c) != 0) {
            isLowerCase = false;
        }
    }
    if (isLowerCase) {
        return LetterCase::Lowercase;
    }
    else if (isUpperCase) {
        return LetterCase::Uppercase;
    }
    else if (isUpperCamelCase) {
        return LetterCase::Titlecase;
    }
    return LetterCase::Lowercase;
}

#if 0
void TestCase_GetLetterCase()
{
    assert(GetLetterCase("Word") == LetterCase::UpperCamelCase);
    assert(GetLetterCase("WORD") == LetterCase::UpperCase);
    assert(GetLetterCase("word") == LetterCase::LowerCase);
}
#endif

void TransformLetterCase(std::u32string & word, LetterCase letterCase)
{
    switch (letterCase) {
    case LetterCase::Lowercase:
        for (auto & c : word) {
            c = ToLowerUtf32(c);
        }
        break;
    case LetterCase::Uppercase:
        for (auto & c : word) {
            c = ToUpperUtf32(c);
        }
        break;
    case LetterCase::Titlecase: {
        for (auto & c : word) {
            c = ToLowerUtf32(c);
        }
        if (!word.empty()) {
            auto & c = word.front();
            c = ToUpperUtf32(c);
        }
        break;
    }
    }
}

template <class SignatureHashing>
SpellCheckResultInternal SuggestLetterCase(
    const std::u32string& word,
    const std::unordered_map<uint32_t, std::vector<std::u32string>>& hashedDictionary)
{
    const auto distanceThreshold = std::min(static_cast<int>(word.size()), 10);
    auto result = SpellCheckBySignatureHashing<SignatureHashing>(
        word,
        hashedDictionary,
        distanceThreshold);

    if (result.suggestions.empty()) {
        return result;
    }

    auto letterCase = GetLetterCase(word);
    for (auto & suggestion : result.suggestions) {
        auto suggestionLetterCase = GetLetterCase(suggestion.word);
        if (suggestionLetterCase == LetterCase::Lowercase) {
            TransformLetterCase(suggestion.word, letterCase);
        }
    }
    return result;
}

template <typename T>
void ResizeSuggestions(T & suggestions, std::size_t elementCount)
{
    if (suggestions.size() > elementCount) {
        suggestions.resize(elementCount);
    }
}

template <class SignatureHashing>
void SeparateWords(
    const std::u32string& word,
    const std::unordered_map<uint32_t, std::vector<std::u32string>>& hashedDictionary,
    std::vector<SpellSuggestion> & suggestions)
{
    constexpr std::size_t prefixMinSize = 2;
    constexpr std::size_t suffixMinSize = 2;
    constexpr std::size_t minSize = 6;
    static_assert(minSize >= prefixMinSize + suffixMinSize, "");

    if ((word.size() < minSize) || (word.size() > 23)) {
        return;
    }

    double maxSimilarity = 0.75;

    const auto end = word.size() - (suffixMinSize - 1);
    for (std::size_t separator = prefixMinSize; separator < end; ++separator) {
        const auto prefix = word.substr(0, separator);
        const auto suffix = word.substr(separator);

        auto prefixSuggestion = ExistWordBySignatureHashing<SignatureHashing>(
            prefix, hashedDictionary);
        if (!prefixSuggestion) {
            continue;
        }
        auto suffixSuggestion = ExistWordBySignatureHashing<SignatureHashing>(
            suffix, hashedDictionary);
        if (!suffixSuggestion) {
            continue;
        }

        assert(prefixSuggestion);
        assert(suffixSuggestion);
        assert(prefixSuggestion->similarity);
        assert(suffixSuggestion->similarity);

        auto similarity = (*prefixSuggestion->similarity + *suffixSuggestion->similarity) / 2.0;
        if (similarity < maxSimilarity) {
            continue;
        }
        maxSimilarity = similarity;

        {
            auto letterCase = GetLetterCase(prefix);
            auto suggestionLetterCase = GetLetterCase(prefixSuggestion->word);
            if (suggestionLetterCase == LetterCase::Lowercase) {
                TransformLetterCase(prefixSuggestion->word, letterCase);
            }
        }
        {
            auto letterCase = GetLetterCase(suffix);
            auto suggestionLetterCase = GetLetterCase(suffixSuggestion->word);
            if (suggestionLetterCase == LetterCase::Lowercase) {
                TransformLetterCase(suffixSuggestion->word, letterCase);
            }
        }

        auto concatWithoutSpace = prefixSuggestion->word + suffixSuggestion->word;
        auto found = std::find_if(std::begin(suggestions), std::end(suggestions), [&](const SpellSuggestion& s) {
            return s.word == concatWithoutSpace;
        });
        if (found != std::end(suggestions)) {
            // NOTE: "accept able" and "acceptable" are duplication.
            continue;
        }

        SpellSuggestion result;
        result.word = prefixSuggestion->word + U" " + suffixSuggestion->word;

        suggestions.push_back(std::move(result));
    }
}

SpellCheckResult ConvertToSpellCheckResult(SpellCheckResultInternal& spellCheck)
{
    SpellCheckResult result;
    result.correctlySpelled = spellCheck.correctlySpelled;
    for (auto & suggestion : spellCheck.suggestions) {
        result.suggestions.push_back(ToUtf8(suggestion.word));
    }
    return result;
}

enum class CharacterType {
    Segmenter,
    Lowercase,
    Uppercase,
};

CharacterType GetCharacterType(char32_t c)
{
    ///@todo Support for other Unicode characters.
    if (u'A' <= c && c <= u'Z') {
        return CharacterType::Uppercase;
    }
    if (u'a' <= c && c <= u'z') {
        return CharacterType::Lowercase;
    }
    if (u'0' <= c && c <= u'9') {
        return CharacterType::Segmenter;
    }
    return CharacterType::Segmenter;
}

struct PartOfIdentifier {
    std::u32string word;
    bool isSegmenter;
};

std::vector<PartOfIdentifier> SplitIdentifier(const std::u32string& text)
{
    std::vector<PartOfIdentifier> words;
    std::u32string wordBuffer;

    auto flush = [&](bool isSegmenter) {
        PartOfIdentifier partOfIdentifier;
        partOfIdentifier.word = wordBuffer;
        partOfIdentifier.isSegmenter = isSegmenter;
        words.push_back(std::move(partOfIdentifier));
        wordBuffer.clear();
    };

    CharacterType startType = CharacterType::Segmenter;

    for (auto iter = std::begin(text); iter != std::end(text); ++iter) {
        const auto c = *iter;
        if (wordBuffer.empty()) {
            wordBuffer += c;
            startType = GetCharacterType(c);
            continue;
        }

        assert(!wordBuffer.empty());
        const auto type = GetCharacterType(c);

        if (startType == CharacterType::Segmenter) {
            if (type == CharacterType::Segmenter) {
                wordBuffer += c;
                continue;
            }
            else {
                flush(true);
                wordBuffer += c;
                startType = type;
                continue;
            }
        }
        else if (startType == CharacterType::Uppercase) {
            if (type == CharacterType::Lowercase) {
                wordBuffer += c;
                continue;
            }
            else if (type == CharacterType::Uppercase) {
                if (GetCharacterType(wordBuffer.back()) == CharacterType::Lowercase) {
                    assert(startType == CharacterType::Uppercase);
                    assert(type == CharacterType::Uppercase);
                    flush(false);
                    wordBuffer += c;
                    startType = type;
                    continue;
                }
            }
            else {
                flush(false);
                wordBuffer += c;
                startType = type;
                continue;
            }
        }

        if (type != GetCharacterType(wordBuffer.back())) {
            flush(startType == CharacterType::Segmenter);
            wordBuffer += c;
            startType = type;
            continue;
        }
        wordBuffer += c;
    }

    if (!wordBuffer.empty()) {
        flush(startType == CharacterType::Segmenter);
    }
    return words;
}

template <class SignatureHashing>
void ParseIdentifier(
    const std::u32string& word,
    const std::unordered_map<uint32_t, std::vector<std::u32string>>& hashedDictionary,
    SpellCheckResultInternal & result)
{
    bool exactMatching = true;
    bool misspellingFound = false;

    auto splitResult = SplitIdentifier(word);
    if (splitResult.size() <= 1) {
        return;
    }

    std::vector<std::u32string> concatSuggestions;
    auto concatenate = [&](const std::u32string& w) {
        if (concatSuggestions.empty()) {
            concatSuggestions.push_back(w);
            return;
        }
        for (auto & concatString : concatSuggestions) {
            concatString += w;
        }
    };

    for (auto & p : splitResult) {
        if (p.isSegmenter) {
            concatenate(p.word);
            continue;
        }
        auto result1 = SuggestLetterCase<SignatureHashing>(p.word, hashedDictionary);
        if (result1.correctlySpelled) {
            concatenate(p.word);
            misspellingFound = true;
            continue;
        }
        exactMatching = false;
        if (result1.suggestions.empty()) {
            concatenate(p.word);
            continue;
        }

        misspellingFound = true;
        SortSuggestions(p.word, result1.suggestions);
        ResizeSuggestions(result1.suggestions, 2);

        // NOTE: Generating combinations
        std::vector<std::u32string> tempSuggestions;
        std::swap(concatSuggestions, tempSuggestions);
        if (tempSuggestions.empty()) {
            tempSuggestions.push_back(U"");
        }
        for (auto & cs : tempSuggestions) {
            for (auto & ss : result1.suggestions) {
                concatSuggestions.push_back(cs + ss.word);
            }
        }
    }

    if (!misspellingFound) {
        return;
    }

    for (auto & cs : concatSuggestions) {
        SpellSuggestion suggestion;
        suggestion.word = cs;
        result.suggestions.push_back(std::move(suggestion));
    }
    result.correctlySpelled = exactMatching;
}

using SignatureHashing = SignatureHashAlphabet;

} // end anonymous namespace

void SpellChecker::AddWord(const std::string& wordUtf8)
{
    auto word = ToUtf32(wordUtf8);
    if (word.empty()) {
        return;
    }

    auto signatureHash = SignatureHashing::Hash(word);
    auto mapIter = hashedDictionary.find(signatureHash);
    if (mapIter == std::end(hashedDictionary)) {
        std::vector<std::u32string> dict;
        hashedDictionary.emplace(signatureHash, std::move(dict));
        mapIter = hashedDictionary.find(signatureHash);
    }
    assert(mapIter != std::end(hashedDictionary));

    auto & words = mapIter->second;
    auto wordsIter = std::lower_bound(std::begin(words), std::end(words), word);
    if ((wordsIter != std::end(words)) && (*wordsIter == word)) {
        // NOTE: The word already exists in a dictionary.
        return;
    }
    words.insert(wordsIter, word);
}

void SpellChecker::RemoveWord(const std::string& wordUtf8)
{
    auto word = ToUtf32(wordUtf8);

    auto signatureHash = SignatureHashing::Hash(word);
    auto mapIter = hashedDictionary.find(signatureHash);
    if (mapIter == std::end(hashedDictionary)) {
        return;
    }
    assert(mapIter != std::end(hashedDictionary));

    auto & words = mapIter->second;
    auto wordsIter = std::lower_bound(std::begin(words), std::end(words), word);
    if ((wordsIter != std::end(words)) && (*wordsIter == word)) {
        words.erase(wordsIter);
    }
}

SpellCheckResult SpellChecker::Suggest(const std::string& wordUtf8)
{
    auto word = ToUtf32(wordUtf8);

    constexpr std::size_t maxSuggestions = 8;

    auto result = SuggestLetterCase<SignatureHashing>(word, hashedDictionary);
    if (result.correctlySpelled) {
        SortSuggestions(word, result.suggestions);
        ResizeSuggestions(result.suggestions, maxSuggestions);
        return ConvertToSpellCheckResult(result);
    }

    ParseIdentifier<SignatureHashing>(word, hashedDictionary, result);

    if (result.correctlySpelled) {
        SortSuggestions(word, result.suggestions);
        ResizeSuggestions(result.suggestions, maxSuggestions);
        return ConvertToSpellCheckResult(result);
    }

    SeparateWords<SignatureHashing>(word, hashedDictionary, result.suggestions);

    SortSuggestions(word, result.suggestions);
    ResizeSuggestions(result.suggestions, maxSuggestions);

    return ConvertToSpellCheckResult(result);
}

std::u32string ToUtf32(const std::string& utf8)
{
    if (utf8.empty()) {
        return {};
    }

#if defined(_MSC_VER) && (_MSC_VER <= 1900)
    std::wstring_convert<std::codecvt_utf8<int32_t>, int32_t> convert;
    auto s = convert.from_bytes(utf8);
    return std::u32string(reinterpret_cast<const char32_t*>(s.data()), s.size());
#else
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    return convert.from_bytes(utf8);
#endif
}

std::string ToUtf8(const std::u32string& utf32)
{
    if (utf32.empty()) {
        return {};
    }

#if defined(_MSC_VER) && (_MSC_VER <= 1900)
    std::wstring_convert<std::codecvt_utf8<int32_t>, int32_t> convert;
    auto p = reinterpret_cast<const int32_t*>(utf32.data());
    return convert.to_bytes(p, p + utf32.size());
#else
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    return convert.to_bytes(utf32);
#endif
}

namespace {

struct UnicodeLetterCaseDesc {
    char32_t character;
    char32_t lowercase;
    char32_t uppercase;
};

// NOTE: UTF-32 encoding table using code points in the U+0000 to U+024F range.
// To reduce memory footprint, we use char16_t instead of char32_t.
constexpr char16_t toLowerCharacters[592] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 0, 0, 0, 0, 0,
    0, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 0, 248, 249, 250, 251, 252, 253, 254, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 0, 248, 249, 250, 251, 252, 253, 254, 255,
    257, 257, 259, 259, 261, 261, 263, 263, 265, 265, 267, 267, 269, 269, 271, 271,
    273, 273, 275, 275, 277, 277, 279, 279, 281, 281, 283, 283, 285, 285, 287, 287,
    289, 289, 291, 291, 293, 293, 295, 295, 297, 297, 299, 299, 301, 301, 303, 303,
    105, 305, 307, 307, 309, 309, 311, 311, 312, 314, 314, 316, 316, 318, 318, 320,
    320, 322, 322, 324, 324, 326, 326, 328, 328, 329, 331, 331, 333, 333, 335, 335,
    337, 337, 339, 339, 341, 341, 343, 343, 345, 345, 347, 347, 349, 349, 351, 351,
    353, 353, 355, 355, 357, 357, 359, 359, 361, 361, 363, 363, 365, 365, 367, 367,
    369, 369, 371, 371, 373, 373, 375, 375, 255, 378, 378, 380, 380, 382, 382, 383,
    384, 595, 387, 387, 389, 389, 596, 392, 392, 598, 599, 396, 396, 397, 477, 601,
    603, 402, 402, 608, 611, 405, 617, 616, 409, 409, 410, 411, 623, 626, 414, 629,
    417, 417, 419, 419, 421, 421, 640, 424, 424, 643, 426, 427, 429, 429, 648, 432,
    432, 650, 651, 436, 436, 438, 438, 658, 441, 441, 442, 0, 445, 445, 446, 447,
    0, 0, 0, 0, 454, 454, 454, 457, 457, 457, 460, 460, 460, 462, 462, 464,
    464, 466, 466, 468, 468, 470, 470, 472, 472, 474, 474, 476, 476, 477, 479, 479,
    481, 481, 483, 483, 485, 485, 487, 487, 489, 489, 491, 491, 493, 493, 495, 495,
    496, 499, 499, 499, 501, 501, 405, 447, 505, 505, 507, 507, 509, 509, 511, 511,
    513, 513, 515, 515, 517, 517, 519, 519, 521, 521, 523, 523, 525, 525, 527, 527,
    529, 529, 531, 531, 533, 533, 535, 535, 537, 537, 539, 539, 541, 541, 543, 543,
    414, 545, 547, 547, 549, 549, 551, 551, 553, 553, 555, 555, 557, 557, 559, 559,
    561, 561, 563, 563, 564, 565, 566, 567, 568, 569, 11365, 572, 572, 410, 11366, 575,
    576, 578, 578, 384, 649, 652, 583, 583, 585, 585, 587, 587, 589, 589, 591, 591,
};

constexpr char16_t toUpperCharacters[592] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 0, 0, 0, 0, 0,
    0, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 924, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 0, 216, 217, 218, 219, 220, 221, 222, 0,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 0, 216, 217, 218, 219, 220, 221, 222, 376,
    256, 256, 258, 258, 260, 260, 262, 262, 264, 264, 266, 266, 268, 268, 270, 270,
    272, 272, 274, 274, 276, 276, 278, 278, 280, 280, 282, 282, 284, 284, 286, 286,
    288, 288, 290, 290, 292, 292, 294, 294, 296, 296, 298, 298, 300, 300, 302, 302,
    304, 73, 306, 306, 308, 308, 310, 310, 0, 313, 313, 315, 315, 317, 317, 319,
    319, 321, 321, 323, 323, 325, 325, 327, 327, 0, 330, 330, 332, 332, 334, 334,
    336, 336, 338, 338, 340, 340, 342, 342, 344, 344, 346, 346, 348, 348, 350, 350,
    352, 352, 354, 354, 356, 356, 358, 358, 360, 360, 362, 362, 364, 364, 366, 366,
    368, 368, 370, 370, 372, 372, 374, 374, 376, 377, 377, 379, 379, 381, 381, 83,
    579, 385, 386, 386, 388, 388, 390, 391, 391, 393, 394, 395, 395, 0, 398, 399,
    400, 401, 401, 403, 404, 502, 406, 407, 408, 408, 573, 0, 412, 413, 544, 415,
    416, 416, 418, 418, 420, 420, 422, 423, 423, 425, 0, 0, 428, 428, 430, 431,
    431, 433, 434, 435, 435, 437, 437, 439, 440, 440, 0, 0, 444, 444, 0, 503,
    0, 0, 0, 0, 452, 452, 452, 455, 455, 455, 458, 458, 458, 461, 461, 463,
    463, 465, 465, 467, 467, 469, 469, 471, 471, 473, 473, 475, 475, 398, 478, 478,
    480, 480, 482, 482, 484, 484, 486, 486, 488, 488, 490, 490, 492, 492, 494, 494,
    0, 497, 497, 497, 500, 500, 502, 503, 504, 504, 506, 506, 508, 508, 510, 510,
    512, 512, 514, 514, 516, 516, 518, 518, 520, 520, 522, 522, 524, 524, 526, 526,
    528, 528, 530, 530, 532, 532, 534, 534, 536, 536, 538, 538, 540, 540, 542, 542,
    544, 0, 546, 546, 548, 548, 550, 550, 552, 552, 554, 554, 556, 556, 558, 558,
    560, 560, 562, 562, 0, 0, 0, 0, 0, 0, 570, 571, 571, 573, 574, 11390,
    11391, 577, 577, 579, 580, 581, 582, 582, 584, 584, 586, 586, 588, 588, 590, 590,
};

// NOTE: The table covers characters/alphabets of the following languages:
// Albanian, Catalan, Croatian, Cyrillic, Czech, Esperanto, Estonian, French,
// German, Greek, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese,
// Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish,
// Turkish.
constexpr UnicodeLetterCaseDesc letterCaseDescs[203] = {
    {0x0370, 0x0371, 0x0370}, {0x0371, 0x0371, 0x0370}, {0x0372, 0x0373, 0x0372},
    {0x0373, 0x0373, 0x0372}, {0x0376, 0x0377, 0x0376}, {0x0377, 0x0377, 0x0376},
    {0x037b, 0x037b, 0x03fd}, {0x037c, 0x037c, 0x03fe}, {0x037d, 0x037d, 0x03ff},
    {0x037f, 0x03f3, 0x037f}, {0x0386, 0x03ac, 0x0386}, {0x0388, 0x03ad, 0x0388},
    {0x0389, 0x03ae, 0x0389}, {0x038a, 0x03af, 0x038a}, {0x038c, 0x03cc, 0x038c},
    {0x038e, 0x03cd, 0x038e}, {0x038f, 0x03ce, 0x038f}, {0x0391, 0x03b1, 0x0391},
    {0x0392, 0x03b2, 0x0392}, {0x0393, 0x03b3, 0x0393}, {0x0394, 0x03b4, 0x0394},
    {0x0395, 0x03b5, 0x0395}, {0x0396, 0x03b6, 0x0396}, {0x0397, 0x03b7, 0x0397},
    {0x0398, 0x03b8, 0x0398}, {0x0399, 0x03b9, 0x0399}, {0x039a, 0x03ba, 0x039a},
    {0x039b, 0x03bb, 0x039b}, {0x039c, 0x03bc, 0x039c}, {0x039d, 0x03bd, 0x039d},
    {0x039e, 0x03be, 0x039e}, {0x039f, 0x03bf, 0x039f}, {0x03a0, 0x03c0, 0x03a0},
    {0x03a1, 0x03c1, 0x03a1}, {0x03a3, 0x03c3, 0x03a3}, {0x03a4, 0x03c4, 0x03a4},
    {0x03a5, 0x03c5, 0x03a5}, {0x03a6, 0x03c6, 0x03a6}, {0x03a7, 0x03c7, 0x03a7},
    {0x03a8, 0x03c8, 0x03a8}, {0x03a9, 0x03c9, 0x03a9}, {0x03aa, 0x03ca, 0x03aa},
    {0x03ab, 0x03cb, 0x03ab}, {0x03ac, 0x03ac, 0x0386}, {0x03ad, 0x03ad, 0x0388},
    {0x03ae, 0x03ae, 0x0389}, {0x03af, 0x03af, 0x038a}, {0x03b1, 0x03b1, 0x0391},
    {0x03b2, 0x03b2, 0x0392}, {0x03b3, 0x03b3, 0x0393}, {0x03b4, 0x03b4, 0x0394},
    {0x03b5, 0x03b5, 0x0395}, {0x03b6, 0x03b6, 0x0396}, {0x03b7, 0x03b7, 0x0397},
    {0x03b8, 0x03b8, 0x0398}, {0x03b9, 0x03b9, 0x0399}, {0x03ba, 0x03ba, 0x039a},
    {0x03bb, 0x03bb, 0x039b}, {0x03bc, 0x03bc, 0x039c}, {0x03bd, 0x03bd, 0x039d},
    {0x03be, 0x03be, 0x039e}, {0x03bf, 0x03bf, 0x039f}, {0x03c0, 0x03c0, 0x03a0},
    {0x03c1, 0x03c1, 0x03a1}, {0x03c2, 0x03c2, 0x03a3}, {0x03c3, 0x03c3, 0x03a3},
    {0x03c4, 0x03c4, 0x03a4}, {0x03c5, 0x03c5, 0x03a5}, {0x03c6, 0x03c6, 0x03a6},
    {0x03c7, 0x03c7, 0x03a7}, {0x03c8, 0x03c8, 0x03a8}, {0x03c9, 0x03c9, 0x03a9},
    {0x03ca, 0x03ca, 0x03aa}, {0x03cb, 0x03cb, 0x03ab}, {0x03cc, 0x03cc, 0x038c},
    {0x03cd, 0x03cd, 0x038e}, {0x03ce, 0x03ce, 0x038f}, {0x03cf, 0x03d7, 0x03cf},
    {0x03d0, 0x03d0, 0x0392}, {0x03d1, 0x03d1, 0x0398}, {0x03d5, 0x03d5, 0x03a6},
    {0x03d6, 0x03d6, 0x03a0}, {0x03d7, 0x03d7, 0x03cf}, {0x03d8, 0x03d9, 0x03d8},
    {0x03d9, 0x03d9, 0x03d8}, {0x03da, 0x03db, 0x03da}, {0x03db, 0x03db, 0x03da},
    {0x03dc, 0x03dd, 0x03dc}, {0x03dd, 0x03dd, 0x03dc}, {0x03de, 0x03df, 0x03de},
    {0x03df, 0x03df, 0x03de}, {0x03e0, 0x03e1, 0x03e0}, {0x03e1, 0x03e1, 0x03e0},
    {0x03e2, 0x03e3, 0x03e2}, {0x03e3, 0x03e3, 0x03e2}, {0x03e4, 0x03e5, 0x03e4},
    {0x03e5, 0x03e5, 0x03e4}, {0x03e6, 0x03e7, 0x03e6}, {0x03e7, 0x03e7, 0x03e6},
    {0x03e8, 0x03e9, 0x03e8}, {0x03e9, 0x03e9, 0x03e8}, {0x03ea, 0x03eb, 0x03ea},
    {0x03eb, 0x03eb, 0x03ea}, {0x03ec, 0x03ed, 0x03ec}, {0x03ed, 0x03ed, 0x03ec},
    {0x03ee, 0x03ef, 0x03ee}, {0x03ef, 0x03ef, 0x03ee}, {0x03f0, 0x03f0, 0x039a},
    {0x03f1, 0x03f1, 0x03a1}, {0x03f2, 0x03f2, 0x03f9}, {0x03f3, 0x03f3, 0x037f},
    {0x03f4, 0x03b8, 0x03f4}, {0x03f5, 0x03f5, 0x0395}, {0x03f7, 0x03f8, 0x03f7},
    {0x03f8, 0x03f8, 0x03f7}, {0x03f9, 0x03f2, 0x03f9}, {0x03fa, 0x03fb, 0x03fa},
    {0x03fb, 0x03fb, 0x03fa}, {0x03fd, 0x037b, 0x03fd}, {0x03fe, 0x037c, 0x03fe},
    {0x03ff, 0x037d, 0x03ff}, {0x0401, 0x0451, 0x0401}, {0x0405, 0x0455, 0x0405},
    {0x0406, 0x0456, 0x0406}, {0x0410, 0x0430, 0x0410}, {0x0411, 0x0431, 0x0411},
    {0x0412, 0x0432, 0x0412}, {0x0413, 0x0433, 0x0413}, {0x0414, 0x0434, 0x0414},
    {0x0415, 0x0435, 0x0415}, {0x0416, 0x0436, 0x0416}, {0x0417, 0x0437, 0x0417},
    {0x0418, 0x0438, 0x0418}, {0x0419, 0x0439, 0x0419}, {0x041a, 0x043a, 0x041a},
    {0x041b, 0x043b, 0x041b}, {0x041c, 0x043c, 0x041c}, {0x041d, 0x043d, 0x041d},
    {0x041e, 0x043e, 0x041e}, {0x041f, 0x043f, 0x041f}, {0x0420, 0x0440, 0x0420},
    {0x0421, 0x0441, 0x0421}, {0x0422, 0x0442, 0x0422}, {0x0423, 0x0443, 0x0423},
    {0x0424, 0x0444, 0x0424}, {0x0425, 0x0445, 0x0425}, {0x0426, 0x0446, 0x0426},
    {0x0427, 0x0447, 0x0427}, {0x0428, 0x0448, 0x0428}, {0x0429, 0x0449, 0x0429},
    {0x042a, 0x044a, 0x042a}, {0x042b, 0x044b, 0x042b}, {0x042c, 0x044c, 0x042c},
    {0x042d, 0x044d, 0x042d}, {0x042e, 0x044e, 0x042e}, {0x042f, 0x044f, 0x042f},
    {0x0430, 0x0430, 0x0410}, {0x0431, 0x0431, 0x0411}, {0x0432, 0x0432, 0x0412},
    {0x0433, 0x0433, 0x0413}, {0x0434, 0x0434, 0x0414}, {0x0435, 0x0435, 0x0415},
    {0x0436, 0x0436, 0x0416}, {0x0437, 0x0437, 0x0417}, {0x0438, 0x0438, 0x0418},
    {0x0439, 0x0439, 0x0419}, {0x043a, 0x043a, 0x041a}, {0x043b, 0x043b, 0x041b},
    {0x043c, 0x043c, 0x041c}, {0x043d, 0x043d, 0x041d}, {0x043e, 0x043e, 0x041e},
    {0x043f, 0x043f, 0x041f}, {0x0440, 0x0440, 0x0420}, {0x0441, 0x0441, 0x0421},
    {0x0442, 0x0442, 0x0422}, {0x0443, 0x0443, 0x0423}, {0x0444, 0x0444, 0x0424},
    {0x0445, 0x0445, 0x0425}, {0x0446, 0x0446, 0x0426}, {0x0447, 0x0447, 0x0427},
    {0x0448, 0x0448, 0x0428}, {0x0449, 0x0449, 0x0429}, {0x044a, 0x044a, 0x042a},
    {0x044b, 0x044b, 0x042b}, {0x044c, 0x044c, 0x042c}, {0x044d, 0x044d, 0x042d},
    {0x044e, 0x044e, 0x042e}, {0x044f, 0x044f, 0x042f}, {0x0451, 0x0451, 0x0401},
    {0x0460, 0x0461, 0x0460}, {0x0462, 0x0463, 0x0462}, {0x0464, 0x0465, 0x0464},
    {0x0466, 0x0467, 0x0466}, {0x0468, 0x0469, 0x0468}, {0x046a, 0x046b, 0x046a},
    {0x046c, 0x046d, 0x046c}, {0x046e, 0x046f, 0x046e}, {0x0470, 0x0471, 0x0470},
    {0x0472, 0x0473, 0x0472}, {0x0474, 0x0475, 0x0474}, {0x1e9e, 0x00df, 0x1e9e},
    {0xa64a, 0xa64b, 0xa64a}, {0xa656, 0xa657, 0xa656}
};

template <typename T, std::size_t N>
constexpr std::size_t SizeOfArray(T (&)[N]) { return N; }

template <typename T>
auto FindCharacterDesc(T& container, char32_t c) -> decltype(std::begin(container))
{
    UnicodeLetterCaseDesc desc;
    desc.character = c;
    auto iter = std::lower_bound(std::begin(container), std::end(container), desc, [](auto & a, auto & b) {
        return a.character < b.character;
    });
    if ((iter != std::end(container)) && (iter->character == c)) {
        return iter;
    }
    return std::end(container);
}

} // end anonymous namespace

bool IsLowerUtf32(char32_t c) noexcept
{
    if (c < SizeOfArray(toLowerCharacters)) {
        auto t = toLowerCharacters[c];
        return (t != 0) && (t == c);
    }
    auto iter = FindCharacterDesc(letterCaseDescs, c);
    if (iter == std::end(letterCaseDescs)) {
        return false;
    }
    return iter->lowercase == c;
}

bool IsUpperUtf32(char32_t c) noexcept
{
    if (c < SizeOfArray(toUpperCharacters)) {
        auto t = toUpperCharacters[c];
        return (t != 0) && (t == c);
    }
    auto iter = FindCharacterDesc(letterCaseDescs, c);
    if (iter == std::end(letterCaseDescs)) {
        return false;
    }
    return iter->uppercase == c;
}

char32_t ToLowerUtf32(char32_t c) noexcept
{
    if (c < SizeOfArray(toLowerCharacters)) {
        auto t = toLowerCharacters[c];
        return (t == 0) ? c : t;
    }
    auto iter = FindCharacterDesc(letterCaseDescs, c);
    if (iter == std::end(letterCaseDescs)) {
        return c;
    }
    return iter->lowercase;
}

char32_t ToUpperUtf32(char32_t c) noexcept
{
    if (c < SizeOfArray(toUpperCharacters)) {
        auto t = toUpperCharacters[c];
        return (t == 0) ? c : t;
    }
    auto iter = FindCharacterDesc(letterCaseDescs, c);
    if (iter == std::end(letterCaseDescs)) {
        return c;
    }
    return iter->uppercase;
}

} // namespace typopoi

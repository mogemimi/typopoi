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

constexpr char32_t toLowerCharacters[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 0, 0, 0, 0, 0,
    0, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 0, 0, 0, 0, 0,
};

constexpr char32_t toUpperCharacters[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 0, 0, 0, 0, 0,
    0, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 0, 0, 0, 0, 0,
};

// NOTE: This table covers characters/alphabets of the following languages:
// Albanian, Catalan, Croatian, Cyrillic, Czech, Esperanto, Estonian, French,
// German, Greek, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Maltese,
// Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish,
// Turkish.
constexpr UnicodeLetterCaseDesc letterCaseDescs[576] = {
    {0x00b5, 0x00b5, 0x039c}, {0x00c0, 0x00e0, 0x00c0}, {0x00c1, 0x00e1, 0x00c1},
    {0x00c2, 0x00e2, 0x00c2}, {0x00c3, 0x00e3, 0x00c3}, {0x00c4, 0x00e4, 0x00c4},
    {0x00c5, 0x00e5, 0x00c5}, {0x00c6, 0x00e6, 0x00c6}, {0x00c7, 0x00e7, 0x00c7},
    {0x00c8, 0x00e8, 0x00c8}, {0x00c9, 0x00e9, 0x00c9}, {0x00ca, 0x00ea, 0x00ca},
    {0x00cb, 0x00eb, 0x00cb}, {0x00cc, 0x00ec, 0x00cc}, {0x00cd, 0x00ed, 0x00cd},
    {0x00ce, 0x00ee, 0x00ce}, {0x00cf, 0x00ef, 0x00cf}, {0x00d0, 0x00f0, 0x00d0},
    {0x00d1, 0x00f1, 0x00d1}, {0x00d2, 0x00f2, 0x00d2}, {0x00d3, 0x00f3, 0x00d3},
    {0x00d4, 0x00f4, 0x00d4}, {0x00d5, 0x00f5, 0x00d5}, {0x00d6, 0x00f6, 0x00d6},
    {0x00d8, 0x00f8, 0x00d8}, {0x00d9, 0x00f9, 0x00d9}, {0x00da, 0x00fa, 0x00da},
    {0x00db, 0x00fb, 0x00db}, {0x00dc, 0x00fc, 0x00dc}, {0x00dd, 0x00fd, 0x00dd},
    {0x00de, 0x00fe, 0x00de}, {0x00e0, 0x00e0, 0x00c0}, {0x00e1, 0x00e1, 0x00c1},
    {0x00e2, 0x00e2, 0x00c2}, {0x00e3, 0x00e3, 0x00c3}, {0x00e4, 0x00e4, 0x00c4},
    {0x00e5, 0x00e5, 0x00c5}, {0x00e6, 0x00e6, 0x00c6}, {0x00e7, 0x00e7, 0x00c7},
    {0x00e8, 0x00e8, 0x00c8}, {0x00e9, 0x00e9, 0x00c9}, {0x00ea, 0x00ea, 0x00ca},
    {0x00eb, 0x00eb, 0x00cb}, {0x00ec, 0x00ec, 0x00cc}, {0x00ed, 0x00ed, 0x00cd},
    {0x00ee, 0x00ee, 0x00ce}, {0x00ef, 0x00ef, 0x00cf}, {0x00f0, 0x00f0, 0x00d0},
    {0x00f1, 0x00f1, 0x00d1}, {0x00f2, 0x00f2, 0x00d2}, {0x00f3, 0x00f3, 0x00d3},
    {0x00f4, 0x00f4, 0x00d4}, {0x00f5, 0x00f5, 0x00d5}, {0x00f6, 0x00f6, 0x00d6},
    {0x00f8, 0x00f8, 0x00d8}, {0x00f9, 0x00f9, 0x00d9}, {0x00fa, 0x00fa, 0x00da},
    {0x00fb, 0x00fb, 0x00db}, {0x00fc, 0x00fc, 0x00dc}, {0x00fd, 0x00fd, 0x00dd},
    {0x00fe, 0x00fe, 0x00de}, {0x00ff, 0x00ff, 0x0178}, {0x0100, 0x0101, 0x0100},
    {0x0101, 0x0101, 0x0100}, {0x0102, 0x0103, 0x0102}, {0x0103, 0x0103, 0x0102},
    {0x0104, 0x0105, 0x0104}, {0x0105, 0x0105, 0x0104}, {0x0106, 0x0107, 0x0106},
    {0x0107, 0x0107, 0x0106}, {0x0108, 0x0109, 0x0108}, {0x0109, 0x0109, 0x0108},
    {0x010a, 0x010b, 0x010a}, {0x010b, 0x010b, 0x010a}, {0x010c, 0x010d, 0x010c},
    {0x010d, 0x010d, 0x010c}, {0x010e, 0x010f, 0x010e}, {0x010f, 0x010f, 0x010e},
    {0x0110, 0x0111, 0x0110}, {0x0111, 0x0111, 0x0110}, {0x0112, 0x0113, 0x0112},
    {0x0113, 0x0113, 0x0112}, {0x0114, 0x0115, 0x0114}, {0x0115, 0x0115, 0x0114},
    {0x0116, 0x0117, 0x0116}, {0x0117, 0x0117, 0x0116}, {0x0118, 0x0119, 0x0118},
    {0x0119, 0x0119, 0x0118}, {0x011a, 0x011b, 0x011a}, {0x011b, 0x011b, 0x011a},
    {0x011c, 0x011d, 0x011c}, {0x011d, 0x011d, 0x011c}, {0x011e, 0x011f, 0x011e},
    {0x011f, 0x011f, 0x011e}, {0x0120, 0x0121, 0x0120}, {0x0121, 0x0121, 0x0120},
    {0x0122, 0x0123, 0x0122}, {0x0123, 0x0123, 0x0122}, {0x0124, 0x0125, 0x0124},
    {0x0125, 0x0125, 0x0124}, {0x0126, 0x0127, 0x0126}, {0x0127, 0x0127, 0x0126},
    {0x0128, 0x0129, 0x0128}, {0x0129, 0x0129, 0x0128}, {0x012a, 0x012b, 0x012a},
    {0x012b, 0x012b, 0x012a}, {0x012c, 0x012d, 0x012c}, {0x012d, 0x012d, 0x012c},
    {0x012e, 0x012f, 0x012e}, {0x012f, 0x012f, 0x012e}, {0x0130, 0x0069, 0x0130},
    {0x0131, 0x0131, 0x0049}, {0x0132, 0x0133, 0x0132}, {0x0133, 0x0133, 0x0132},
    {0x0134, 0x0135, 0x0134}, {0x0135, 0x0135, 0x0134}, {0x0136, 0x0137, 0x0136},
    {0x0137, 0x0137, 0x0136}, {0x0139, 0x013a, 0x0139}, {0x013a, 0x013a, 0x0139},
    {0x013b, 0x013c, 0x013b}, {0x013c, 0x013c, 0x013b}, {0x013d, 0x013e, 0x013d},
    {0x013e, 0x013e, 0x013d}, {0x013f, 0x0140, 0x013f}, {0x0140, 0x0140, 0x013f},
    {0x0141, 0x0142, 0x0141}, {0x0142, 0x0142, 0x0141}, {0x0143, 0x0144, 0x0143},
    {0x0144, 0x0144, 0x0143}, {0x0145, 0x0146, 0x0145}, {0x0146, 0x0146, 0x0145},
    {0x0147, 0x0148, 0x0147}, {0x0148, 0x0148, 0x0147}, {0x014a, 0x014b, 0x014a},
    {0x014b, 0x014b, 0x014a}, {0x014c, 0x014d, 0x014c}, {0x014d, 0x014d, 0x014c},
    {0x014e, 0x014f, 0x014e}, {0x014f, 0x014f, 0x014e}, {0x0150, 0x0151, 0x0150},
    {0x0151, 0x0151, 0x0150}, {0x0152, 0x0153, 0x0152}, {0x0153, 0x0153, 0x0152},
    {0x0154, 0x0155, 0x0154}, {0x0155, 0x0155, 0x0154}, {0x0156, 0x0157, 0x0156},
    {0x0157, 0x0157, 0x0156}, {0x0158, 0x0159, 0x0158}, {0x0159, 0x0159, 0x0158},
    {0x015a, 0x015b, 0x015a}, {0x015b, 0x015b, 0x015a}, {0x015c, 0x015d, 0x015c},
    {0x015d, 0x015d, 0x015c}, {0x015e, 0x015f, 0x015e}, {0x015f, 0x015f, 0x015e},
    {0x0160, 0x0161, 0x0160}, {0x0161, 0x0161, 0x0160}, {0x0162, 0x0163, 0x0162},
    {0x0163, 0x0163, 0x0162}, {0x0164, 0x0165, 0x0164}, {0x0165, 0x0165, 0x0164},
    {0x0166, 0x0167, 0x0166}, {0x0167, 0x0167, 0x0166}, {0x0168, 0x0169, 0x0168},
    {0x0169, 0x0169, 0x0168}, {0x016a, 0x016b, 0x016a}, {0x016b, 0x016b, 0x016a},
    {0x016c, 0x016d, 0x016c}, {0x016d, 0x016d, 0x016c}, {0x016e, 0x016f, 0x016e},
    {0x016f, 0x016f, 0x016e}, {0x0170, 0x0171, 0x0170}, {0x0171, 0x0171, 0x0170},
    {0x0172, 0x0173, 0x0172}, {0x0173, 0x0173, 0x0172}, {0x0174, 0x0175, 0x0174},
    {0x0175, 0x0175, 0x0174}, {0x0176, 0x0177, 0x0176}, {0x0177, 0x0177, 0x0176},
    {0x0178, 0x00ff, 0x0178}, {0x0179, 0x017a, 0x0179}, {0x017a, 0x017a, 0x0179},
    {0x017b, 0x017c, 0x017b}, {0x017c, 0x017c, 0x017b}, {0x017d, 0x017e, 0x017d},
    {0x017e, 0x017e, 0x017d}, {0x017f, 0x017f, 0x0053}, {0x0180, 0x0180, 0x0243},
    {0x0181, 0x0253, 0x0181}, {0x0182, 0x0183, 0x0182}, {0x0183, 0x0183, 0x0182},
    {0x0184, 0x0185, 0x0184}, {0x0185, 0x0185, 0x0184}, {0x0186, 0x0254, 0x0186},
    {0x0187, 0x0188, 0x0187}, {0x0188, 0x0188, 0x0187}, {0x0189, 0x0256, 0x0189},
    {0x018a, 0x0257, 0x018a}, {0x018b, 0x018c, 0x018b}, {0x018c, 0x018c, 0x018b},
    {0x018e, 0x01dd, 0x018e}, {0x018f, 0x0259, 0x018f}, {0x0190, 0x025b, 0x0190},
    {0x0191, 0x0192, 0x0191}, {0x0192, 0x0192, 0x0191}, {0x0193, 0x0260, 0x0193},
    {0x0194, 0x0263, 0x0194}, {0x0195, 0x0195, 0x01f6}, {0x0196, 0x0269, 0x0196},
    {0x0197, 0x0268, 0x0197}, {0x0198, 0x0199, 0x0198}, {0x0199, 0x0199, 0x0198},
    {0x019a, 0x019a, 0x023d}, {0x019c, 0x026f, 0x019c}, {0x019d, 0x0272, 0x019d},
    {0x019e, 0x019e, 0x0220}, {0x019f, 0x0275, 0x019f}, {0x01a0, 0x01a1, 0x01a0},
    {0x01a1, 0x01a1, 0x01a0}, {0x01a2, 0x01a3, 0x01a2}, {0x01a3, 0x01a3, 0x01a2},
    {0x01a4, 0x01a5, 0x01a4}, {0x01a5, 0x01a5, 0x01a4}, {0x01a6, 0x0280, 0x01a6},
    {0x01a7, 0x01a8, 0x01a7}, {0x01a8, 0x01a8, 0x01a7}, {0x01a9, 0x0283, 0x01a9},
    {0x01ac, 0x01ad, 0x01ac}, {0x01ad, 0x01ad, 0x01ac}, {0x01ae, 0x0288, 0x01ae},
    {0x01af, 0x01b0, 0x01af}, {0x01b0, 0x01b0, 0x01af}, {0x01b1, 0x028a, 0x01b1},
    {0x01b2, 0x028b, 0x01b2}, {0x01b3, 0x01b4, 0x01b3}, {0x01b4, 0x01b4, 0x01b3},
    {0x01b5, 0x01b6, 0x01b5}, {0x01b6, 0x01b6, 0x01b5}, {0x01b7, 0x0292, 0x01b7},
    {0x01b8, 0x01b9, 0x01b8}, {0x01b9, 0x01b9, 0x01b8}, {0x01bc, 0x01bd, 0x01bc},
    {0x01bd, 0x01bd, 0x01bc}, {0x01bf, 0x01bf, 0x01f7}, {0x01c4, 0x01c6, 0x01c4},
    {0x01c6, 0x01c6, 0x01c4}, {0x01c7, 0x01c9, 0x01c7}, {0x01c9, 0x01c9, 0x01c7},
    {0x01ca, 0x01cc, 0x01ca}, {0x01cc, 0x01cc, 0x01ca}, {0x01cd, 0x01ce, 0x01cd},
    {0x01ce, 0x01ce, 0x01cd}, {0x01cf, 0x01d0, 0x01cf}, {0x01d0, 0x01d0, 0x01cf},
    {0x01d1, 0x01d2, 0x01d1}, {0x01d2, 0x01d2, 0x01d1}, {0x01d3, 0x01d4, 0x01d3},
    {0x01d4, 0x01d4, 0x01d3}, {0x01d5, 0x01d6, 0x01d5}, {0x01d6, 0x01d6, 0x01d5},
    {0x01d7, 0x01d8, 0x01d7}, {0x01d8, 0x01d8, 0x01d7}, {0x01d9, 0x01da, 0x01d9},
    {0x01da, 0x01da, 0x01d9}, {0x01db, 0x01dc, 0x01db}, {0x01dc, 0x01dc, 0x01db},
    {0x01dd, 0x01dd, 0x018e}, {0x01de, 0x01df, 0x01de}, {0x01df, 0x01df, 0x01de},
    {0x01e0, 0x01e1, 0x01e0}, {0x01e1, 0x01e1, 0x01e0}, {0x01e2, 0x01e3, 0x01e2},
    {0x01e3, 0x01e3, 0x01e2}, {0x01e4, 0x01e5, 0x01e4}, {0x01e5, 0x01e5, 0x01e4},
    {0x01e6, 0x01e7, 0x01e6}, {0x01e7, 0x01e7, 0x01e6}, {0x01e8, 0x01e9, 0x01e8},
    {0x01e9, 0x01e9, 0x01e8}, {0x01ea, 0x01eb, 0x01ea}, {0x01eb, 0x01eb, 0x01ea},
    {0x01ec, 0x01ed, 0x01ec}, {0x01ed, 0x01ed, 0x01ec}, {0x01ee, 0x01ef, 0x01ee},
    {0x01ef, 0x01ef, 0x01ee}, {0x01f1, 0x01f3, 0x01f1}, {0x01f3, 0x01f3, 0x01f1},
    {0x01f4, 0x01f5, 0x01f4}, {0x01f5, 0x01f5, 0x01f4}, {0x01f6, 0x0195, 0x01f6},
    {0x01f7, 0x01bf, 0x01f7}, {0x01f8, 0x01f9, 0x01f8}, {0x01f9, 0x01f9, 0x01f8},
    {0x01fa, 0x01fb, 0x01fa}, {0x01fb, 0x01fb, 0x01fa}, {0x01fc, 0x01fd, 0x01fc},
    {0x01fd, 0x01fd, 0x01fc}, {0x01fe, 0x01ff, 0x01fe}, {0x01ff, 0x01ff, 0x01fe},
    {0x0200, 0x0201, 0x0200}, {0x0201, 0x0201, 0x0200}, {0x0202, 0x0203, 0x0202},
    {0x0203, 0x0203, 0x0202}, {0x0204, 0x0205, 0x0204}, {0x0205, 0x0205, 0x0204},
    {0x0206, 0x0207, 0x0206}, {0x0207, 0x0207, 0x0206}, {0x0208, 0x0209, 0x0208},
    {0x0209, 0x0209, 0x0208}, {0x020a, 0x020b, 0x020a}, {0x020b, 0x020b, 0x020a},
    {0x020c, 0x020d, 0x020c}, {0x020d, 0x020d, 0x020c}, {0x020e, 0x020f, 0x020e},
    {0x020f, 0x020f, 0x020e}, {0x0210, 0x0211, 0x0210}, {0x0211, 0x0211, 0x0210},
    {0x0212, 0x0213, 0x0212}, {0x0213, 0x0213, 0x0212}, {0x0214, 0x0215, 0x0214},
    {0x0215, 0x0215, 0x0214}, {0x0216, 0x0217, 0x0216}, {0x0217, 0x0217, 0x0216},
    {0x0218, 0x0219, 0x0218}, {0x0219, 0x0219, 0x0218}, {0x021a, 0x021b, 0x021a},
    {0x021b, 0x021b, 0x021a}, {0x021c, 0x021d, 0x021c}, {0x021d, 0x021d, 0x021c},
    {0x021e, 0x021f, 0x021e}, {0x021f, 0x021f, 0x021e}, {0x0220, 0x019e, 0x0220},
    {0x0222, 0x0223, 0x0222}, {0x0223, 0x0223, 0x0222}, {0x0224, 0x0225, 0x0224},
    {0x0225, 0x0225, 0x0224}, {0x0226, 0x0227, 0x0226}, {0x0227, 0x0227, 0x0226},
    {0x0228, 0x0229, 0x0228}, {0x0229, 0x0229, 0x0228}, {0x022a, 0x022b, 0x022a},
    {0x022b, 0x022b, 0x022a}, {0x022c, 0x022d, 0x022c}, {0x022d, 0x022d, 0x022c},
    {0x022e, 0x022f, 0x022e}, {0x022f, 0x022f, 0x022e}, {0x0230, 0x0231, 0x0230},
    {0x0231, 0x0231, 0x0230}, {0x0232, 0x0233, 0x0232}, {0x0233, 0x0233, 0x0232},
    {0x023a, 0x2c65, 0x023a}, {0x023b, 0x023c, 0x023b}, {0x023c, 0x023c, 0x023b},
    {0x023d, 0x019a, 0x023d}, {0x023e, 0x2c66, 0x023e}, {0x023f, 0x023f, 0x2c7e},
    {0x0240, 0x0240, 0x2c7f}, {0x0241, 0x0242, 0x0241}, {0x0242, 0x0242, 0x0241},
    {0x0243, 0x0180, 0x0243}, {0x0244, 0x0289, 0x0244}, {0x0245, 0x028c, 0x0245},
    {0x0246, 0x0247, 0x0246}, {0x0247, 0x0247, 0x0246}, {0x0248, 0x0249, 0x0248},
    {0x0249, 0x0249, 0x0248}, {0x024a, 0x024b, 0x024a}, {0x024b, 0x024b, 0x024a},
    {0x024c, 0x024d, 0x024c}, {0x024d, 0x024d, 0x024c}, {0x024e, 0x024f, 0x024e},
    {0x024f, 0x024f, 0x024e}, {0x0370, 0x0371, 0x0370}, {0x0371, 0x0371, 0x0370},
    {0x0372, 0x0373, 0x0372}, {0x0373, 0x0373, 0x0372}, {0x0376, 0x0377, 0x0376},
    {0x0377, 0x0377, 0x0376}, {0x037b, 0x037b, 0x03fd}, {0x037c, 0x037c, 0x03fe},
    {0x037d, 0x037d, 0x03ff}, {0x037f, 0x03f3, 0x037f}, {0x0386, 0x03ac, 0x0386},
    {0x0388, 0x03ad, 0x0388}, {0x0389, 0x03ae, 0x0389}, {0x038a, 0x03af, 0x038a},
    {0x038c, 0x03cc, 0x038c}, {0x038e, 0x03cd, 0x038e}, {0x038f, 0x03ce, 0x038f},
    {0x0391, 0x03b1, 0x0391}, {0x0392, 0x03b2, 0x0392}, {0x0393, 0x03b3, 0x0393},
    {0x0394, 0x03b4, 0x0394}, {0x0395, 0x03b5, 0x0395}, {0x0396, 0x03b6, 0x0396},
    {0x0397, 0x03b7, 0x0397}, {0x0398, 0x03b8, 0x0398}, {0x0399, 0x03b9, 0x0399},
    {0x039a, 0x03ba, 0x039a}, {0x039b, 0x03bb, 0x039b}, {0x039c, 0x03bc, 0x039c},
    {0x039d, 0x03bd, 0x039d}, {0x039e, 0x03be, 0x039e}, {0x039f, 0x03bf, 0x039f},
    {0x03a0, 0x03c0, 0x03a0}, {0x03a1, 0x03c1, 0x03a1}, {0x03a3, 0x03c3, 0x03a3},
    {0x03a4, 0x03c4, 0x03a4}, {0x03a5, 0x03c5, 0x03a5}, {0x03a6, 0x03c6, 0x03a6},
    {0x03a7, 0x03c7, 0x03a7}, {0x03a8, 0x03c8, 0x03a8}, {0x03a9, 0x03c9, 0x03a9},
    {0x03aa, 0x03ca, 0x03aa}, {0x03ab, 0x03cb, 0x03ab}, {0x03ac, 0x03ac, 0x0386},
    {0x03ad, 0x03ad, 0x0388}, {0x03ae, 0x03ae, 0x0389}, {0x03af, 0x03af, 0x038a},
    {0x03b1, 0x03b1, 0x0391}, {0x03b2, 0x03b2, 0x0392}, {0x03b3, 0x03b3, 0x0393},
    {0x03b4, 0x03b4, 0x0394}, {0x03b5, 0x03b5, 0x0395}, {0x03b6, 0x03b6, 0x0396},
    {0x03b7, 0x03b7, 0x0397}, {0x03b8, 0x03b8, 0x0398}, {0x03b9, 0x03b9, 0x0399},
    {0x03ba, 0x03ba, 0x039a}, {0x03bb, 0x03bb, 0x039b}, {0x03bc, 0x03bc, 0x039c},
    {0x03bd, 0x03bd, 0x039d}, {0x03be, 0x03be, 0x039e}, {0x03bf, 0x03bf, 0x039f},
    {0x03c0, 0x03c0, 0x03a0}, {0x03c1, 0x03c1, 0x03a1}, {0x03c2, 0x03c2, 0x03a3},
    {0x03c3, 0x03c3, 0x03a3}, {0x03c4, 0x03c4, 0x03a4}, {0x03c5, 0x03c5, 0x03a5},
    {0x03c6, 0x03c6, 0x03a6}, {0x03c7, 0x03c7, 0x03a7}, {0x03c8, 0x03c8, 0x03a8},
    {0x03c9, 0x03c9, 0x03a9}, {0x03ca, 0x03ca, 0x03aa}, {0x03cb, 0x03cb, 0x03ab},
    {0x03cc, 0x03cc, 0x038c}, {0x03cd, 0x03cd, 0x038e}, {0x03ce, 0x03ce, 0x038f},
    {0x03cf, 0x03d7, 0x03cf}, {0x03d0, 0x03d0, 0x0392}, {0x03d1, 0x03d1, 0x0398},
    {0x03d5, 0x03d5, 0x03a6}, {0x03d6, 0x03d6, 0x03a0}, {0x03d7, 0x03d7, 0x03cf},
    {0x03d8, 0x03d9, 0x03d8}, {0x03d9, 0x03d9, 0x03d8}, {0x03da, 0x03db, 0x03da},
    {0x03db, 0x03db, 0x03da}, {0x03dc, 0x03dd, 0x03dc}, {0x03dd, 0x03dd, 0x03dc},
    {0x03de, 0x03df, 0x03de}, {0x03df, 0x03df, 0x03de}, {0x03e0, 0x03e1, 0x03e0},
    {0x03e1, 0x03e1, 0x03e0}, {0x03e2, 0x03e3, 0x03e2}, {0x03e3, 0x03e3, 0x03e2},
    {0x03e4, 0x03e5, 0x03e4}, {0x03e5, 0x03e5, 0x03e4}, {0x03e6, 0x03e7, 0x03e6},
    {0x03e7, 0x03e7, 0x03e6}, {0x03e8, 0x03e9, 0x03e8}, {0x03e9, 0x03e9, 0x03e8},
    {0x03ea, 0x03eb, 0x03ea}, {0x03eb, 0x03eb, 0x03ea}, {0x03ec, 0x03ed, 0x03ec},
    {0x03ed, 0x03ed, 0x03ec}, {0x03ee, 0x03ef, 0x03ee}, {0x03ef, 0x03ef, 0x03ee},
    {0x03f0, 0x03f0, 0x039a}, {0x03f1, 0x03f1, 0x03a1}, {0x03f2, 0x03f2, 0x03f9},
    {0x03f3, 0x03f3, 0x037f}, {0x03f4, 0x03b8, 0x03f4}, {0x03f5, 0x03f5, 0x0395},
    {0x03f7, 0x03f8, 0x03f7}, {0x03f8, 0x03f8, 0x03f7}, {0x03f9, 0x03f2, 0x03f9},
    {0x03fa, 0x03fb, 0x03fa}, {0x03fb, 0x03fb, 0x03fa}, {0x03fd, 0x037b, 0x03fd},
    {0x03fe, 0x037c, 0x03fe}, {0x03ff, 0x037d, 0x03ff}, {0x0401, 0x0451, 0x0401},
    {0x0405, 0x0455, 0x0405}, {0x0406, 0x0456, 0x0406}, {0x0410, 0x0430, 0x0410},
    {0x0411, 0x0431, 0x0411}, {0x0412, 0x0432, 0x0412}, {0x0413, 0x0433, 0x0413},
    {0x0414, 0x0434, 0x0414}, {0x0415, 0x0435, 0x0415}, {0x0416, 0x0436, 0x0416},
    {0x0417, 0x0437, 0x0417}, {0x0418, 0x0438, 0x0418}, {0x0419, 0x0439, 0x0419},
    {0x041a, 0x043a, 0x041a}, {0x041b, 0x043b, 0x041b}, {0x041c, 0x043c, 0x041c},
    {0x041d, 0x043d, 0x041d}, {0x041e, 0x043e, 0x041e}, {0x041f, 0x043f, 0x041f},
    {0x0420, 0x0440, 0x0420}, {0x0421, 0x0441, 0x0421}, {0x0422, 0x0442, 0x0422},
    {0x0423, 0x0443, 0x0423}, {0x0424, 0x0444, 0x0424}, {0x0425, 0x0445, 0x0425},
    {0x0426, 0x0446, 0x0426}, {0x0427, 0x0447, 0x0427}, {0x0428, 0x0448, 0x0428},
    {0x0429, 0x0449, 0x0429}, {0x042a, 0x044a, 0x042a}, {0x042b, 0x044b, 0x042b},
    {0x042c, 0x044c, 0x042c}, {0x042d, 0x044d, 0x042d}, {0x042e, 0x044e, 0x042e},
    {0x042f, 0x044f, 0x042f}, {0x0430, 0x0430, 0x0410}, {0x0431, 0x0431, 0x0411},
    {0x0432, 0x0432, 0x0412}, {0x0433, 0x0433, 0x0413}, {0x0434, 0x0434, 0x0414},
    {0x0435, 0x0435, 0x0415}, {0x0436, 0x0436, 0x0416}, {0x0437, 0x0437, 0x0417},
    {0x0438, 0x0438, 0x0418}, {0x0439, 0x0439, 0x0419}, {0x043a, 0x043a, 0x041a},
    {0x043b, 0x043b, 0x041b}, {0x043c, 0x043c, 0x041c}, {0x043d, 0x043d, 0x041d},
    {0x043e, 0x043e, 0x041e}, {0x043f, 0x043f, 0x041f}, {0x0440, 0x0440, 0x0420},
    {0x0441, 0x0441, 0x0421}, {0x0442, 0x0442, 0x0422}, {0x0443, 0x0443, 0x0423},
    {0x0444, 0x0444, 0x0424}, {0x0445, 0x0445, 0x0425}, {0x0446, 0x0446, 0x0426},
    {0x0447, 0x0447, 0x0427}, {0x0448, 0x0448, 0x0428}, {0x0449, 0x0449, 0x0429},
    {0x044a, 0x044a, 0x042a}, {0x044b, 0x044b, 0x042b}, {0x044c, 0x044c, 0x042c},
    {0x044d, 0x044d, 0x042d}, {0x044e, 0x044e, 0x042e}, {0x044f, 0x044f, 0x042f},
    {0x0451, 0x0451, 0x0401}, {0x0460, 0x0461, 0x0460}, {0x0462, 0x0463, 0x0462},
    {0x0464, 0x0465, 0x0464}, {0x0466, 0x0467, 0x0466}, {0x0468, 0x0469, 0x0468},
    {0x046a, 0x046b, 0x046a}, {0x046c, 0x046d, 0x046c}, {0x046e, 0x046f, 0x046e},
    {0x0470, 0x0471, 0x0470}, {0x0472, 0x0473, 0x0472}, {0x0474, 0x0475, 0x0474},
    {0x1e9e, 0x00df, 0x1e9e}, {0xa64a, 0xa64b, 0xa64a}, {0xa656, 0xa657, 0xa656},
};

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
    if (c < 128) {
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
    if (c < 128) {
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
    if (c < 128) {
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
    if (c < 128) {
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

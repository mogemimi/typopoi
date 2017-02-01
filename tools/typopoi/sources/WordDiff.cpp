// Copyright (c) 2015 mogemimi. Distributed under the MIT license.

#include "WordDiff.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <utility>
#include <experimental/optional>

namespace typopoi {
namespace {

void CompressHunks(std::vector<DiffHunk<char>> & hunks)
{
    std::vector<DiffHunk<char>> oldHunks;
    std::swap(hunks, oldHunks);

    for (auto & hunk : oldHunks) {
        if (hunks.empty() || (hunks.back().operation != hunk.operation)) {
            hunks.push_back(std::move(hunk));
        }
        else {
            assert(hunks.back().operation == hunk.operation);
            hunks.back().text += hunk.text;
        }
    }
}

template <typename T>
using Optional = std::experimental::optional<T>;

} // end anonymous namespace

std::vector<DiffHunk<char>> computeDiff(const std::string& text1, const std::string& text2)
{
    return computeDiff_ONDGreedyAlgorithm(text1, text2);
}

namespace {

struct Vertex final {
    std::shared_ptr<Vertex> prev;
    int x;
    int y;

    Vertex(std::shared_ptr<Vertex> && prevIn, int xIn, int yIn)
        : prev(std::move(prevIn))
        , x(xIn)
        , y(yIn)
    {
    }

    void PushBack(int xIn, int yIn)
    {
        auto vertex = std::make_shared<Vertex>(std::move(prev), x, y);
        prev = std::move(vertex);
        x = xIn;
        y = yIn;
    }
};

std::vector<DiffHunk<char>> GenerateDiffHunks(
    const std::shared_ptr<Vertex>& path,
    const std::string& text1,
    const std::string& text2)
{
    std::vector<DiffHunk<char>> hunks;

    std::vector<Vertex> points;
    auto iter = path;
    while (iter != nullptr) {
        points.push_back(*iter);
        iter = iter->prev;
    }
    std::reverse(std::begin(points), std::end(points));

    for (size_t i = 0; (i + 1) < points.size(); ++i) {
        auto & p0 = points[i];
        auto & p1 = points[i + 1];
        assert(((p0.x == p1.x) && (p0.y + 1 == p1.y))
            || ((p0.x + 1 == p1.x) && (p0.y == p1.y))
            || ((p0.x + 1 == p1.x) && (p0.y + 1 == p1.y)));
        if (p0.x == p1.x) {
            DiffHunk<char> hunk;
            hunk.operation = DiffOperation::Insertion;
            hunk.text = text2[p0.y];
            hunks.push_back(std::move(hunk));
        }
        else if (p0.y == p1.y) {
            DiffHunk<char> hunk;
            hunk.operation = DiffOperation::Deletion;
            hunk.text = text1[p0.x];
            hunks.push_back(std::move(hunk));
        }
        else {
            DiffHunk<char> hunk;
            hunk.operation = DiffOperation::Equality;
            hunk.text = text1[p0.x];
            hunks.push_back(std::move(hunk));
        }
    }

    CompressHunks(hunks);
    return hunks;
}

} // end anonymous namespace

std::vector<DiffHunk<char>> computeDiff_ONDGreedyAlgorithm(const std::string& text1, const std::string& text2)
{
    // NOTE:
    // This algorithm is based on Myers's An O((M+N)D) Greedy Algorithm in
    // "An O(ND)Difference Algorithm and Its Variations",
    // Algorithmica (1986), pages 251-266.

    if (text1.empty() || text2.empty()) {
        std::vector<DiffHunk<char>> hunks;
        if (!text1.empty()) {
            DiffHunk<char> hunk1;
            hunk1.operation = DiffOperation::Deletion;
            hunk1.text = text1;
            hunks.push_back(std::move(hunk1));
        }
        else if (!text2.empty()) {
            DiffHunk<char> hunk2;
            hunk2.operation = DiffOperation::Insertion;
            hunk2.text = text2;
            hunks.push_back(std::move(hunk2));
        }
        return hunks;
    }

    const auto M = static_cast<int>(text1.size());
    const auto N = static_cast<int>(text2.size());

    const auto maxD = M + N;
    const auto offset = N;

    std::vector<int> vertices(M + N + 1);
    vertices[1 + offset] = 0;

    std::vector<std::shared_ptr<Vertex>> paths(vertices.size());

    for (int d = 0; d <= maxD; ++d) {
        const int startK = -std::min(d, (N * 2) - d);
        const int endK = std::min(d, (M * 2) - d);

        assert((-N <= startK) && (endK <= M));
        assert(std::abs(startK % 2) == (d % 2));
        assert(std::abs(endK % 2) == (d % 2));
        assert((d > N) ? (startK == -(N * 2 - d)) : (startK == -d));
        assert((d > M) ? (endK == (M * 2 - d)) : (endK == d));

        // NOTE:
        // When IMPROVE_DIFFHUNK_READABILITY is defined 1,
        // * Diff(aa, bb) is -aa+bb
        // * Diff(a, bbbbb) is -a+bbbbb
        // * Diff(aaaaa, bbbbb) is -aaaaa+bbbbb
        //
        // When IMPROVE_DIFFHUNK_READABILITY is defined 0,
        // * Diff(aa, bb) is -a+b-a+b
        // * Diff(a, bbbbb) is +b-a+bbbb
        // * Diff(aaaaa, bbbbb) is -aaaa+b-a+bbbb
#define IMPROVE_DIFFHUNK_READABILITY 1
#if (IMPROVE_DIFFHUNK_READABILITY == 1)
        const bool startKMinusOneEnabled = (-d < startK) && (-N < startK);
        const bool endKMinusOneEnabled = (endK < d) && (endK < M);
#else
        constexpr bool startKMinusOneEnabled = false;
        constexpr bool endKMinusOneEnabled = false;
#endif

        for (int k = startK; k <= endK; k += 2) {
            assert((-N <= k) && (k <= M));
            assert(std::abs(k % 2) == (d % 2));

            const auto kOffset = k + offset;

            int x = 0;
            if (k == startK && !startKMinusOneEnabled) {
                // NOTE: Move directly from vertex(x, y - 1) to vertex(x, y)
                // NOTE: In this case, V[k - 1] is out of range.
                x = vertices[kOffset + 1];
                paths[kOffset] = paths[kOffset + 1];
            }
            else if (k == endK && !endKMinusOneEnabled) {
                // NOTE: Move directly from vertex(x - 1, y) to vertex(x, y)
                // NOTE: In this case, V[k + 1] is out of range.
                x = vertices[kOffset - 1] + 1;
                paths[kOffset] = paths[kOffset - 1];
            }
            else if (vertices[kOffset - 1] < vertices[kOffset + 1]) {
                // NOTE: Move from vertex(k + 1) to vertex(k)
                // vertex(k + 1) is ahead of vertex(k - 1).
                assert(-N < k && k < M);
                assert((k != -d) && (k != -N));
                assert((k != d) && (k != M));
                x = vertices[kOffset + 1];
                paths[kOffset] = paths[kOffset + 1];
            }
            else {
                // NOTE: Move from vertex(k - 1) to vertex(k)
                // vertex(k - 1) is ahead of vertex(k + 1).
                assert(-N < k && k < M);
                assert((k != -d) && (k != -N));
                assert((k != d) && (k != M));
                assert(vertices[kOffset - 1] >= vertices[kOffset + 1]);
                x = vertices[kOffset - 1] + 1;
                paths[kOffset] = paths[kOffset - 1];
            }

            // NOTE: `k` is defined from `x - y = k`.
            int y = x - k;
            assert(x >= 0 && y >= 0);

#if !defined(NDEBUG)
            if (d == 0) {
                assert((x == 0) && (y == 0) && (k == 0));
                assert(paths[kOffset] == nullptr);
            }
#endif

            auto prev = std::move(paths[kOffset]);
            paths[kOffset] = std::make_shared<Vertex>(std::move(prev), x, y);

            while (x < M && y < N && text1[x] == text2[y]) {
                // NOTE: This loop finds a possibly empty sequence
                // of diagonal edges called a 'snake'.
                x += 1;
                y += 1;
                assert(paths[kOffset] != nullptr);
                paths[kOffset]->PushBack(x, y);
            }

            vertices[kOffset] = x;
            if (x >= M && y >= N) {
                return GenerateDiffHunks(paths[kOffset], text1, text2);
            }
        }
    }

    // NOTE: In this case, D must be == M + N.
    assert(false);
    return {};
}

} // namespace typopoi

// Copyright (c) 2015 mogemimi. Distributed under the MIT license.

#pragma once

#include <string>
#include <experimental/optional>

namespace typopoi {

enum class TerminalColor {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White
};

std::string changeTerminalTextColor(
    const std::string& text,
    TerminalColor textColor,
    std::experimental::optional<TerminalColor> backgroundColor = std::experimental::nullopt);

} // namespace typopoi

#ifdef _MSC_VER
#pragma warning(disable: 4146 4127 4244 4702 4996)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Parse/Parser.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif

#include <iostream>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory MyToolCategory("my-tool options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nMore help text...");

namespace {

class MyPrinter final {
private:
    std::string previousFilePath;

public:
    void Print(const std::string& s, const std::string& filePath)
    {
        if (previousFilePath != filePath) {
            if (!previousFilePath.empty()) {
                std::cout << std::endl;
            }
            std::cout << filePath << ":" << std::endl;
            previousFilePath = filePath;
        }
        std::cout << "    " << s << std::endl;
    }
};

class MyCommentHandler final : public clang::CommentHandler {
private:
    llvm::StringRef filePath;
    MyPrinter & printer;

public:
    explicit MyCommentHandler(MyPrinter & printerIn)
        : printer(printerIn)
    {
    }

    void SetFile(llvm::StringRef filePathIn)
    {
        filePath = filePathIn;
    }

    bool HandleComment(clang::Preprocessor & pp, clang::SourceRange range) override
    {
        clang::SourceManager& sm = pp.getSourceManager();
        if (sm.getFilename(range.getBegin()) != filePath) {
            return false;
        }

        const auto startLoc = sm.getDecomposedLoc(range.getBegin());
        const auto endLoc = sm.getDecomposedLoc(range.getEnd());
        const auto fileData = sm.getBufferData(startLoc.first);

        auto comment = fileData.substr(startLoc.second, endLoc.second - startLoc.second).str();
        printer.Print(comment, filePath);

        return false;
    }
};

class MyASTConsumer final : public ASTConsumer {
};

class MyFrontendAction final : public ASTFrontendAction {
public:
    explicit MyFrontendAction(MyPrinter & printerIn)
        : commentHandler(printerIn)
    {
    }

    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance & ci, StringRef file) override
    {
        commentHandler.SetFile(file);
        ci.getPreprocessor().addCommentHandler(&commentHandler);
        return std::make_unique<MyASTConsumer>();
    }

private:
    MyCommentHandler commentHandler;
};

class MyDiagnosticConsumer final : public clang::DiagnosticConsumer {
public:
    bool IncludeInDiagnosticCounts() const override
    {
        return false;
    }
};

class MyFrontendActionFactory final : public FrontendActionFactory {
public:
    explicit MyFrontendActionFactory(MyPrinter & printerIn)
        : printer(printerIn)
    {
    }

    clang::FrontendAction* create() override
    {
        return new MyFrontendAction(printer);
    }

private:
    MyPrinter & printer;
};

} // unnamed namespace

int main(int argc, const char **argv)
{
    CommonOptionsParser options(argc, argv, MyToolCategory);
    ClangTool tool(options.getCompilations(), options.getSourcePathList());

    MyDiagnosticConsumer diagnosticConsumer;
    tool.setDiagnosticConsumer(&diagnosticConsumer);

    MyPrinter printer;
    return tool.run(std::make_unique<MyFrontendActionFactory>(printer).get());
}

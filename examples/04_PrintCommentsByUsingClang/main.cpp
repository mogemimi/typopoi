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
public:
    void Print(const std::string& s)
    {
        std::cout << s << std::endl;
    }
};

class MyCommentHandler final : public clang::CommentHandler {
private:
    llvm::StringRef inputFile;
    MyPrinter & printer = nullptr;

public:
    void SetFile(llvm::StringRef fileIn)
    {
        inputFile = fileIn;
    }

    void SetPrinter(MyPrinter* printerIn)
    {
        printer = printerIn;
    }

    bool HandleComment(clang::Preprocessor& pp, clang::SourceRange range) override
    {
        clang::SourceManager& sm = pp.getSourceManager();
        if (sm.getFilename(range.getBegin()) != inputFile) {
            return false;
        }
        assert(sm.getFilename(range.getBegin()) == inputFile);

        const auto startLoc = sm.getDecomposedLoc(range.getBegin());
        const auto endLoc = sm.getDecomposedLoc(range.getEnd());
        const auto fileData = sm.getBufferData(startLoc.first);

        assert(printer != nullptr);
        auto sourceString = fileData.substr(startLoc.second, endLoc.second - startLoc.second).str();
        std::cout << inputFile << std::endl;
        std::cout << sourceString << std::endl;

        return false;
    }
};

class MyASTConsumer final : public ASTConsumer {
public:
    explicit MyASTConsumer(MyPrinter & printerIn)
        : printer(printerIn)
    {
    }

    void HandleTranslationUnit(ASTContext& context) override
    {
        matcher.matchAST(context);
    }

    bool HandleTopLevelDecl(clang::DeclGroupRef d) override
    {
        for (auto & iter : d) {
            auto fd = llvm::dyn_cast<clang::NamedDecl>(iter);
            if (fd) {
                std::string identifier = fd->getDeclName().getAsString();
                printer.Print(identifier);
            }
        }
        return true;
    }

private:
    MatchFinder matcher;
    MyPrinter & printer;
};

class MyFrontendAction final : public ASTFrontendAction {
public:
    explicit MyFrontendAction(MyPrinter & printerIn)
        : printer(printerIn)
    {
    }

    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance & ci, StringRef file) override
    {
        commentHandler.SetFile(file);
        commentHandler.SetPrinter(&printer);
        ci.getPreprocessor().addCommentHandler(&commentHandler);
        return std::make_unique<MyASTConsumer>(printer);
    }

private:
    MyPrinter & printer;
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

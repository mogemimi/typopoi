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

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Host.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Frontend/CompilerInstance.h"

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

} // unnamed namespace

int main(int argc, const char **argv)
{
    CommonOptionsParser options(argc, argv, MyToolCategory);

    clang::DiagnosticOptions diagnosticOptions;
    clang::TextDiagnosticPrinter *pTextDiagnosticPrinter =
        new clang::TextDiagnosticPrinter(
            llvm::outs(),
            &diagnosticOptions);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> pDiagIDs;

    clang::DiagnosticsEngine *pDiagnosticsEngine =
        new clang::DiagnosticsEngine(pDiagIDs,
            &diagnosticOptions,
            pTextDiagnosticPrinter);

    clang::LangOptions languageOptions;
    clang::FileSystemOptions fileSystemOptions;
    clang::FileManager fileManager(fileSystemOptions);

    clang::SourceManager sourceManager(
        *pDiagnosticsEngine,
        fileManager);

    std::shared_ptr<clang::TargetOptions> targetOptions = std::make_shared<clang::TargetOptions>();
    //clang::TargetOptions targetOptions;
    targetOptions->Triple = llvm::sys::getDefaultTargetTriple();

    clang::TargetInfo *pTargetInfo =
        clang::TargetInfo::CreateTargetInfo(
            *pDiagnosticsEngine,
            targetOptions);

    llvm::IntrusiveRefCntPtr<clang::HeaderSearchOptions> hso;
    clang::HeaderSearch headerSearch(hso,
                                     sourceManager,
                                     *pDiagnosticsEngine,
                                     languageOptions,
                                     pTargetInfo);
    clang::CompilerInstance compInst;

    llvm::IntrusiveRefCntPtr<clang::PreprocessorOptions> pOpts;

    clang::Preprocessor preprocessor(
        pOpts,
        *pDiagnosticsEngine,
        languageOptions,
        sourceManager,
        headerSearch,
        compInst);

    // // TODO
    // //const clang::FileEntry file = sourceManager.getMainFileID();
    // auto sourceFilePath = options.getSourcePathList().front();
    // const clang::FileEntry *pFile = fileManager.getFile(sourceFilePath);
    //
    // sourceManager.setMainFileID( sourceManager.createFileID( pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
    // preprocessor.EnterMainSourceFile();
    // pTextDiagnosticPrinter->BeginSourceFile(languageOptions, &preprocessor);
    //
    // clang::Token token;
    // do {
    //     preprocessor.Lex(token);
    //     if( pDiagnosticsEngine->hasErrorOccurred())
    //     {
    //         break;
    //     }
    //     preprocessor.DumpToken(token);
    //     std::cerr << std::endl;
    // } while( token.isNot(clang::tok::eof));
    // pTextDiagnosticPrinter->EndSourceFile();

    return 0;
}

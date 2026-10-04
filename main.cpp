#include "app_runner.h"

#include <cctype>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
std::string trimAscii(std::string value) {
    std::string compact;
    compact.reserve(value.size());
    for (char c : value) {
        if (c != '\0') {
            compact.push_back(c);
        }
    }
    value = compact;

    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }

    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }

    if (start > 0) {
        value.erase(0, start);
    }
    return value;
}
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::cout << "\n========================================" << std::endl;
    std::cout << "     LLM from Scratch - Main Menu" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "Choose an application:" << std::endl;
    std::cout << "  1. Tokenizer Demo" << std::endl;
    std::cout << "  2. Compression Demo" << std::endl;
    std::cout << "  3. LLM Model Demo" << std::endl;
    std::cout << "  4. Train Chat Model on Real Data (n-gram)" << std::endl;
    std::cout << "  5. LLM Chat" << std::endl;
    std::cout << "  6. Evaluate Chat Model" << std::endl;
    std::cout << "  7. Train Transformer on Real Corpus (C++/CUDA)" << std::endl;
    std::cout << "\nEnter choice [1-7] (default=3): ";

    std::string choice;
    std::getline(std::cin, choice);
    choice = trimAscii(choice);

    if (choice.empty()) {
        choice = "3";
    }

    int result = 1;
    if (choice == "1") {
        result = runTokenizerApplication();
    } else if (choice == "2") {
        result = runCompressionExample();
    } else if (choice == "3") {
        result = runLLMExample();
    } else if (choice == "4") {
        result = runRealCorpusTrainingExample();
    } else if (choice == "5") {
        result = runLLMChatExample();
    } else if (choice == "6") {
        result = runChatModelEvaluationExample();
    } else if (choice == "7") {
        result = runTransformerCorpusTrainingExample();
    } else {
        std::cerr << "Invalid choice. Enter a number from 1 to 7." << std::endl;
    }

    return result;
}

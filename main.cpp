#include "app_runner.h"

#include <cctype>
#include <iostream>
#include <string>

#ifdef _WIN32
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
    std::cout << "  4. Train Chat Model on Real Data" << std::endl;
    std::cout << "  5. LLM Chat" << std::endl;
    std::cout << "  6. Evaluate Chat Model" << std::endl;
    std::cout << "\nEnter choice [1-6] (default=3): ";

    std::string choice;
    std::getline(std::cin, choice);
    choice = trimAscii(choice);

    if (choice.empty()) {
        choice = "3";
    }

    int result = 1;
    if (choice.find('1') != std::string::npos) {
        result = runTokenizerApplication();
    } else if (choice.find('2') != std::string::npos) {
        result = runCompressionExample();
    } else if (choice.find('3') != std::string::npos) {
        result = runLLMExample();
    } else if (choice.find('4') != std::string::npos) {
        result = runRealCorpusTrainingExample();
    } else if (choice.find('5') != std::string::npos) {
        result = runLLMChatExample();
    } else if (choice.find('6') != std::string::npos) {
        result = runChatModelEvaluationExample();
    } else {
        std::cerr << "Invalid choice." << std::endl;
    }

    return result;
}

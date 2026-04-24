#include "app_runner.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║     LLM from Scratch - Main Menu      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;

    std::cout << "Choose an application:" << std::endl;
    std::cout << "  1. Tokenizer Demo" << std::endl;
    std::cout << "  2. Compression Demo" << std::endl;
    std::cout << "  3. LLM Model Demo" << std::endl;
    std::cout << "\nEnter choice [1-3] (default=3): ";

    std::string choice;
    std::getline(std::cin, choice);
    
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
    } else {
        std::cerr << "Invalid choice." << std::endl;
    }

    return result;
}

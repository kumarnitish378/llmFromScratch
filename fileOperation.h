#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

class FileOperation {
public:
    void writeToFile(const string& filename, const string& content) {
        ofstream outFile(filename);
        if (outFile.is_open()) {
            outFile << content;
            outFile.close();
            cout << "Content written to " << filename << endl;
        } else {
            cerr << "Unable to open file: " << filename << endl;
        }
    }

    string readFromFile(const string& filename) {
        ifstream inFile(filename);
        string content;
        if (inFile.is_open()) {
            getline(inFile, content, '\0'); // Read entire file
            inFile.close();
            cout << "Content read from " << filename << endl;
        } else {
            cerr << "Unable to open file: " << filename << endl;
        }
        return content;
    }

    // function to tokenize the content of the file
    void tokenizeContent(const string& content) {
        istringstream iss(content);
        string token;
        while (iss >> token) {
            // cout << "Token: " << token << endl;
        }
    }

    // function to count the number of lines in the file
    void countLines(const string& filename) {
        ifstream inFile(filename);
        int lineCount = 0;
        string line;
        if (inFile.is_open())
        {
            while (getline(inFile, line)) {
                lineCount++;
            }
            inFile.close();
            cout << "Number of lines in " << filename << ": " << lineCount << endl;
        }
        else
        {
            cerr << "Unable to open file: " << filename << endl;
        }
    }

    // function to get a token by index
    string getTokenByIndex(const string& content, int index) {
        istringstream iss(content);
        string token;
        int currentIndex = 0;
        while (iss >> token) {
            if (currentIndex == index) {
                return token;
            }
            currentIndex++;
        }
        return ""; // Return empty string if index is out of bounds
    }
};
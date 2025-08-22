#include "parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

// Private implementation class
class PTXParser::Impl {
public:
    Impl() {
        // Initialize default state
        m_errorMessage = "";
    }
    
    ~Impl() = default;

    // Parse a PTX file
    bool parseFile(const std::string& filename) {
        // Open file
        std::ifstream file(filename);
        if (!file.is_open()) {
            m_errorMessage = "Failed to open file: " + filename;
            return false;
        }
        
        // Read file contents
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        
        // Parse contents
        return parseString(buffer.str());
    }

    // Parse PTX code from string
    bool parseString(const std::string& ptxCode) {
        // Clear previous instructions
        m_instructions.clear();
        m_errorMessage = "";
        
        // Split code into lines
        std::istringstream iss(ptxCode);
        std::string line;
        size_t lineNumber = 0;
        
        while (std::getline(iss, line)) {
            lineNumber++;
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || 
                (line.length() >= 2 && line[0] == '/' && line[1] == '/')) {
                continue;
            }
            
            // Trim whitespace
            line = trim(line);
            
            // Skip if still empty
            if (line.empty()) {
                continue;
            }
            
            // Create a basic decoded instruction from the line
            DecodedInstruction instruction = {};
            // For now, we just store the raw line - in a real implementation,
            // this would be more sophisticated parsing
            m_instructions.push_back(instruction);
        }
        
        return true;
    }

    // Get parsed instructions
    const std::vector<DecodedInstruction>& getInstructions() const {
        return m_instructions;
    }

    // Get error message if parsing failed
    const std::string& getErrorMessage() const {
        return m_errorMessage;
    }

private:
    // Helper function to trim whitespace
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(' ');
        if (first == std::string::npos) {
            return "";
        }
        
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }
    
    // Parsed instructions
    std::vector<DecodedInstruction> m_instructions;
    
    // Error message
    std::string m_errorMessage;
};

PTXParser::PTXParser() : pImpl(std::make_unique<Impl>()) {}

PTXParser::~PTXParser() = default;

bool PTXParser::parseFile(const std::string& filename) {
    return pImpl->parseFile(filename);
}

bool PTXParser::parseString(const std::string& ptxCode) {
    return pImpl->parseString(ptxCode);
}

const std::vector<DecodedInstruction>& PTXParser::getInstructions() const {
    return pImpl->getInstructions();
}

const std::string& PTXParser::getErrorMessage() const {
    return pImpl->getErrorMessage();
}

// Factory functions
extern "C" {
    PTXParser* createPTXParser() {
        return new PTXParser();
    }
    
    void destroyPTXParser(PTXParser* parser) {
        delete parser;
    }
}
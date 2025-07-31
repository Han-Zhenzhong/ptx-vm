#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>
#include <memory>
#include "include/instruction_types.hpp"

class PTXParser {
public:
    // Constructor/destructor
    PTXParser();
    ~PTXParser();

    // Parse a PTX file
    bool parseFile(const std::string& filename);

    // Parse PTX code from string
    bool parseString(const std::string& ptxCode);

    // Get parsed instructions
    const std::vector<DecodedInstruction>& getInstructions() const;

    // Get error message if parsing failed
    const std::string& getErrorMessage() const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // PARSER_HPP
#include <iostream>
#include <string>

std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, last - first + 1);
}

int main() {
    std::string inner = "%r0+4";
    size_t plusPos = inner.find('+');
    
    std::cout << "inner = '" << inner << "'" << std::endl;
    std::cout << "plusPos = " << plusPos << std::endl;
    
    std::string baseReg = trim(inner.substr(0, plusPos));
    std::string offsetStr = trim(inner.substr(plusPos + 1));
    
    std::cout << "baseReg = '" << baseReg << "'" << std::endl;
    std::cout << "offsetStr = '" << offsetStr << "'" << std::endl;
    
    // Parse numPart
    std::string numPart;
    for (size_t i = 1; i < baseReg.size(); ++i)
    {
        if (std::isdigit(baseReg[i]))
        {
            numPart += baseReg[i];
        }
    }
    std::cout << "numPart = '" << numPart << "'" << std::endl;
    
    if (!numPart.empty()) {
        int regIndex = std::stoi(numPart);
        std::cout << "registerIndex = " << regIndex << std::endl;
    }
    
    return 0;
}

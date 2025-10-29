#ifndef CLI_INTERFACE_HPP
#define CLI_INTERFACE_HPP

#include <string>
#include <vector>
#include "vm.hpp"

class CLIInterface {
public:
    // Constructor/destructor
    CLIInterface();
    ~CLIInterface();

    // Run the command-line interface
    int run(int argc, char* argv[]);

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // CLI_INTERFACE_HPP
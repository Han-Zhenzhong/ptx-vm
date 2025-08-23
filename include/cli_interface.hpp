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

    // Initialize the CLI interface
    bool initialize();

    // Run the command-line interface
    int run(int argc, char* argv[]);

    // Parse and execute a command
    int executeCommand(const std::string& command, const std::vector<std::string>& args);

    // Help command - display available commands
    void helpCommand(const std::vector<std::string>& args);

    // Load command - load a PTX program
    void loadCommand(const std::vector<std::string>& args);

    // Run command - execute the loaded program
    void runCommand(const std::vector<std::string>& args);

    // Step command - execute one instruction
    void stepCommand(const std::vector<std::string>& args);

    // Break command - set a breakpoint
    void breakCommand(const std::vector<std::string>& args);

    // Watch command - set a watchpoint
    void watchCommand(const std::vector<std::string>& args);

    // Register command - display/register information
    void registerCommand(const std::vector<std::string>& args);

    // Memory command - display memory information
    void memoryCommand(const std::vector<std::string>& args);

    // Alloc command - allocate memory
    void allocCommand(const std::vector<std::string>& args);

    // Memcpy command - copy memory
    void memcpyCommand(const std::vector<std::string>& args);

    // Launch command - launch a kernel with parameters
    void launchCommand(const std::vector<std::string>& args);

    // Profiling command - control profiling
    void profileCommand(const std::vector<std::string>& args);

    // Dump command - dump execution statistics
    void dumpCommand(const std::vector<std::string>& args);

    // List command - list loaded program
    void listCommand(const std::vector<std::string>& args);

    // Quit command - exit the VM
    void quitCommand(const std::vector<std::string>& args);

    // Display prompt and get user input
    std::string getCommandLine();

    // Print message to console
    void printMessage(const std::string& message);

    // Print error message to console
    void printError(const std::string& message);

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // CLI_INTERFACE_HPP
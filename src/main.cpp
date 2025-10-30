#include <iostream>
#include <string>
#include "cli_interface.hpp"
#include "logger.hpp"

int main(int argc, char* argv[]) {
    try {
        // Create CLI interface
        CLIInterface cli;
        
        // Run the CLI interface
        return cli.run(argc, argv);
    } catch (const std::exception& e) {
        Logger::error(std::string("Error: ") + e.what());
        return 1;
    } catch (...) {
        Logger::error("Unknown error occurred");
        return 1;
    }
}
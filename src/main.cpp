#include <iostream>
#include <string>
#include "host/cli_interface.hpp"

int main(int argc, char* argv[]) {
    try {
        // Create CLI interface
        CLIInterface cli;
        
        // Run the CLI interface
        return cli.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}
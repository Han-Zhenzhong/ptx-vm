#include "cli_interface.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <cstring>

// Private implementation class
class CLIInterface::Impl {
public:
    Impl() {
        // Initialize VM
        m_vm = std::make_unique<PTXVM>();
        if (!m_vm->initialize()) {
            std::cerr << "Failed to initialize PTX VM" << std::endl;
            throw std::runtime_error("Failed to initialize PTX VM");
        }
    }
    
    ~Impl() = default;

    // Run the command-line interface
    int run(int argc, char* argv[]) {
        // Process command line arguments
        processArguments(argc, argv);
        
        // Main command loop
        bool running = true;
        while (running) {
            // Display prompt and get command
            std::string commandLine = getCommandLine();
            
            // Check for empty command line
            if (commandLine.empty()) {
                continue;
            }
            
            // Parse command line
            std::vector<std::string> tokens = parseCommandLine(commandLine);
            
            // Extract command and arguments
            std::string command = tokens[0];
            std::vector<std::string> args(tokens.begin() + 1, tokens.end());
            
            // Execute command
            if (executeCommand(command, args)) {
                running = false;
            }
        }
        
        return 0;
    }

    // Process command line arguments
    void processArguments(int argc, char* argv[]) {
        // Check if a program was specified on command line
        if (argc > 1) {
            // Load the specified program
            loadProgram(argv[1]);
            
            // If there are more arguments, display warning about correct usage
            if (argc > 2) {
                printMessage("");
                printMessage("=================================================================");
                printMessage("NOTICE: Additional command-line arguments are ignored.");
                printMessage("=================================================================");
                printMessage("");
                printMessage("PTX parameters are automatically typed from kernel signature!");
                printMessage("");
                printMessage("Two types of parameters:");
                printMessage("  1. POINTER (.u64) - device memory addresses (need 'alloc')");
                printMessage("  2. SCALAR (.u32, .f32, etc.) - direct values (no alloc)");
                printMessage("");
                printMessage("Correct workflow:");
                printMessage("  1. For pointer params:       alloc <size>");
                printMessage("  2. Fill pointer data:        fill <address> <count> <values...>");
                printMessage("  3. Launch with mixed params: launch <kernel> <param1> <param2> ...");
                printMessage("");
                printMessage("Example 1 - Pointers only:");
                printMessage("  > alloc 32                    # Returns 0x10000");
                printMessage("  > fill 0x10000 8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0");
                printMessage("  > alloc 32                    # Returns 0x10020");
                printMessage("  > launch vecAdd 0x10000 0x10020");
                printMessage("");
                printMessage("Example 2 - Mixed (pointers + scalars):");
                printMessage("  > alloc 4096                  # Returns 0x10000");
                printMessage("  > fill 0x10000 ...");
                printMessage("  > launch scaleArray 0x10000 1024 2.5");
                printMessage("     kernel signature: (.param .u64 ptr, .param .u32 N, .param .f32 scale)");
                printMessage("     Auto-converts:    0x10000→ptr, 1024→u32, 2.5→f32");
                printMessage("");
                printMessage("See: docs/ptx_entry_function_complete_guide.md");
                printMessage("=================================================================");
                printMessage("");
            }
        } else {
            printMessage("No program specified. Use 'load <filename>' to load a PTX program.");
            printMessage("");
            printMessage("After loading, use interactive commands to:");
            printMessage("  - Allocate memory: alloc <size>");
            printMessage("  - Fill data: fill <address> <count> <values...>");
            printMessage("  - Launch kernel: launch <kernel_name> <addr1> <addr2> ...");
            printMessage("");
            printMessage("Type 'help' for more information.");
        }
    }

    // Parse command line into tokens
    std::vector<std::string> parseCommandLine(const std::string& commandLine) {
        std::vector<std::string> tokens;
        std::istringstream iss(commandLine);
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        return tokens;
    }

    // Execute a command
    bool executeCommand(const std::string& command, const std::vector<std::string>& args) {
        // Convert command to lowercase
        std::string cmd = command;
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);
        
        // Dispatch to appropriate command handler
        if (cmd == "help" || cmd == "?") {
            helpCommand(args);
        } else if (cmd == "load") {
            loadCommand(args);
        } else if (cmd == "alloc") {
            allocCommand(args);
        } else if (cmd == "memcpy") {
            memcpyCommand(args);
        } else if (cmd == "write") {
            writeCommand(args);
        } else if (cmd == "fill") {
            fillCommand(args);
        } else if (cmd == "memory" || cmd == "mem") {
            memoryCommand(args);
        } else if (cmd == "launch") {
            launchCommand(args);
        } else if (cmd == "break" || cmd == "b") {
            breakCommand(args);
        } else if (cmd == "watch" || cmd == "w") {
            watchCommand(args);
        } else if (cmd == "quit" || cmd == "exit" || cmd == "q") {
            quitCommand(args);
            return true; // Exit the CLI
        } else if (cmd == "clear" || cmd == "cls") {
            clearCommand(args);
        } else if (cmd == "version") {
            versionCommand(args);
        } else if (cmd == "info") {
            infoCommand(args);
        } else if (cmd == "visualize" || cmd == "vis") {
            visualizeCommand(args);
        } else {
            std::ostringstream oss;
            oss << "Unknown command: " << command << ". Type 'help' for available commands.";
            printError(oss.str());
        }
        
        return false; // Don't exit the CLI
    }

    // Help command - display available commands
    void helpCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            printMessage("Available commands:");
            printMessage("  help (?, h)              - Show this help message");
            printMessage("  load <filename>         - Load a PTX program");
            printMessage("  alloc <size>           - Allocate memory");
            printMessage("  memcpy <dest> <src> <size> - Copy memory");
            printMessage("  write <address> <value> - Write a value to memory");
            printMessage("  fill <address> <count> <value1> [value2] ... - Fill memory with values");
            printMessage("  memory (mem) <address> <size> - View memory contents");
            printMessage("  launch <kernel> <params...> - Launch kernel (auto-detects param types from PTX)");
            printMessage("  break (b) <address>    - Set a breakpoint");
            printMessage("  watch (w) <address>    - Set a watchpoint");
            printMessage("  visualize (vis) <type>  - Display visualizations (warp, memory, performance)");
            printMessage("  quit (exit, q)         - Quit the VM");
            printMessage("  clear (cls)            - Clear the screen");
            printMessage("  version                 - Show version information");
            printMessage("  info                    - Show current VM information");
            printMessage("");
            printMessage("For detailed help on a specific command, use 'help <command>'.");
        } else {
            // Detailed help for specific command
            std::string cmd = args[0];
            std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);
            
            if (cmd == "help") {
                printMessage("help [command] - Show help for all commands or a specific command");
            } else if (cmd == "load") {
                printMessage("load <filename> - Load a PTX program from disk");
                printMessage("Example: load examples/simple_math_example.ptx");
            } else if (cmd == "alloc") {
                printMessage("alloc <size> - Allocate memory in the VM");
                printMessage("Example: alloc 1024");
            } else if (cmd == "memcpy") {
                printMessage("memcpy <dest> <src> <size> - Copy memory in the VM");
                printMessage("Example: memcpy 0x2000 0x1000 256");
            } else if (cmd == "write") {
                printMessage("write <address> <value> - Write a value to memory");
                printMessage("Example: write 0x10000 42");
                printMessage("This command writes a single byte value to the specified memory address.");
            } else if (cmd == "fill") {
                printMessage("fill <address> <count> <value1> [value2] ... - Fill memory with values");
                printMessage("Example: fill 0x10000 4 1 2 3 4");
                printMessage("This command writes multiple byte values starting at the specified address.");
            } else if (cmd == "memory" || cmd == "mem") {
                printMessage("memory <address> <size> - View memory contents");
                printMessage("Example: memory 0x10000 20");
                printMessage("This command displays memory in hex dump format with ASCII representation.");
                printMessage("If size is a multiple of 4, it also shows decoded int32 values.");
            } else if (cmd == "launch") {
                printMessage("launch <kernel_name> [param1] [param2] ... - Launch a kernel");
                printMessage("");
                printMessage("AUTOMATIC TYPE DETECTION:");
                printMessage("Parameters are automatically typed based on the PTX kernel signature.");
                printMessage("  - No parameters: Kernel accepts no arguments");
                printMessage("  - Pointer types (.u64): Pass device address (e.g., 0x10000)");
                printMessage("  - Scalar types (.u32, .f32, etc.): Pass value directly");
                printMessage("");
                printMessage("Example 0 - No parameters:");
                printMessage("  .entry noArgKernel()");
                printMessage("  > launch noArgKernel");
                printMessage("  (No memory allocation needed)");
                printMessage("");
                printMessage("Example 1 - Pointers only (vector addition):");
                printMessage("  .entry vecAdd(.param .u64 A, .param .u64 B, .param .u64 C)");
                printMessage("  > alloc 32");
                printMessage("    Returns: 0x10000");
                printMessage("  > alloc 32");
                printMessage("    Returns: 0x10020");
                printMessage("  > alloc 32");
                printMessage("    Returns: 0x10040");
                printMessage("  > fill 0x10000 8 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0");
                printMessage("  > launch vecAdd 0x10000 0x10020 0x10040");
                printMessage("");
                printMessage("Example 2 - Mixed pointers and scalars (array scaling):");
                printMessage("  .entry scaleArray(.param .u64 data, .param .u32 N, .param .f32 scale)");
                printMessage("  > alloc 4096");
                printMessage("    Returns: 0x10000");
                printMessage("  > fill 0x10000 1024 ...");
                printMessage("  > launch scaleArray 0x10000 1024 2.5");
                printMessage("    Parameter [0]: device address 0x10000 (pointer)");
                printMessage("    Parameter [1]: value 1024 (scalar .u32)");
                printMessage("    Parameter [2]: value 2.5 (scalar .f32)");
                printMessage("");
                printMessage("Example 3 - Pure scalars (computation):");
                printMessage("  .entry compute(.param .u32 a, .param .u32 b, .param .f32 factor)");
                printMessage("  > launch compute 100 200 1.5");
                printMessage("");
                printMessage("The VM reads the PTX kernel signature and automatically converts");
                printMessage("string arguments to the correct type (u32, f32, u64 pointer, etc.).");
            } else if (cmd == "break" || cmd == "b") {
                printMessage("break <address> - Set a breakpoint at the specified address");
                printMessage("Example: break 0x100");
            } else if (cmd == "watch" || cmd == "w") {
                printMessage("watch <address> - Set a watchpoint at the specified address");
                printMessage("Example: watch 0x1000");
            } else if (cmd == "visualize" || cmd == "vis") {
                printMessage("visualize <type> - Display visualizations");
                printMessage("Types:");
                printMessage("  warp        - Warp execution visualization");
                printMessage("  memory      - Memory access visualization");
                printMessage("  performance - Performance counter display");
                printMessage("Examples:");
                printMessage("  visualize warp");
                printMessage("  visualize memory");
                printMessage("  visualize performance");
            } else if (cmd == "quit" || cmd == "exit" || cmd == "q") {
                printMessage("quit - Exit the virtual machine");
            } else if (cmd == "clear" || cmd == "cls") {
                printMessage("clear - Clear the screen");
            } else if (cmd == "version") {
                printMessage("version - Show version information");
            } else if (cmd == "info") {
                printMessage("info - Show current VM information");
            } else {
                std::ostringstream oss;
                oss << "No detailed help available for command: " << cmd;
                printMessage(oss.str());
            }
        }
    }

    // Load command - load a PTX program
    void loadCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            printError("Usage: load <filename>");
            return;
        }
        
        // Get filename from arguments
        std::string filename = args[0];
        
        // Load and execute the program
        if (m_vm->loadProgram(filename)) {
            m_loadedProgram = filename;
            printMessage("Program loaded successfully.");
            
            // Reset execution state
            resetExecutionState();
        } else {
            printError("Failed to load program. Use 'help load' for usage.");
        }
    }

    // Load a program
    void loadProgram(const std::string& filename) {
        // Load and execute the program
        if (m_vm->loadProgram(filename)) {
            m_loadedProgram = filename;
            
            // Reset execution state
            resetExecutionState();
        }
    }

    // Reset execution state
    void resetExecutionState() {
        m_currentPC = 0;
        m_executing = false;
    }

    // Break command - set a breakpoint
    void breakCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            printError("Usage: break <address>");
            return;
        }
        
        try {
            // Parse address
            uint64_t address = std::stoull(args[0], nullptr, 0);
            
            // Set breakpoint
            bool result = m_vm->getDebugger().setBreakpoint(address);
            
            if (result) {
                std::ostringstream oss;
                oss << "Breakpoint set at address 0x" << std::hex << address << std::dec;
                printMessage(oss.str());
            } else {
                printError("Failed to set breakpoint.");
            }
        } catch (...) {
            printError("Invalid address format. Use hexadecimal (e.g., 0x100).");
        }
    }

    // Watch command - set a watchpoint
    void watchCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            printError("Usage: watch <address>");
            return;
        }
        
        try {
            // Parse address
            uint64_t address = std::stoull(args[0], nullptr, 0);
            
            // Set watchpoint
            bool result = m_vm->setWatchpoint(address);
            
            if (result) {
                std::ostringstream oss;
                oss << "Watchpoint set at address 0x" << std::hex << address << std::dec;
                printMessage(oss.str());
            } else {
                printError("Failed to set watchpoint.");
            }
        } catch (...) {
            printError("Invalid address format. Use hexadecimal (e.g., 0x1000).");
        }
    }

    // Alloc command - allocate memory
    void allocCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            printError("Usage: alloc <size>");
            return;
        }
        
        try {
            // Parse size
            size_t size = std::stoull(args[0], nullptr, 0);
            
            if (size == 0) {
                printError("Size must be greater than 0.");
                return;
            }
            
            if (size > 1024 * 1024) { // 1MB limit
                printError("Size must be less than 1MB.");
                return;
            }
            
            // Call the VM's memory allocation function
            // Simple allocation strategy: track allocations manually
            static uint64_t allocationOffset = 0x10000; // Start at 64KB
            
            MemorySubsystem& memSys = m_vm->getMemorySubsystem();
            size_t globalMemSize = memSys.getMemorySize(MemorySpace::GLOBAL);
            
            if (allocationOffset + size > globalMemSize) {
                printError("Out of memory.");
                return;
            }
            
            CUdeviceptr ptr = allocationOffset;
            allocationOffset += size;
            // Ensure 8-byte alignment
            allocationOffset = (allocationOffset + 7) & ~7;
            
            std::ostringstream oss;
            oss << "Allocated " << size << " bytes at address 0x" << std::hex << ptr << std::dec;
            printMessage(oss.str());
        } catch (...) {
            printError("Invalid size format.");
        }
    }

    // Memcpy command - copy memory
    void memcpyCommand(const std::vector<std::string>& args) {
        if (args.size() < 3) {
            printError("Usage: memcpy <dest> <src> <size>");
            printError("Example: memcpy 0x2000 0x1000 256");
            return;
        }
        
        try {
            // Parse destination address
            uint64_t dest = std::stoull(args[0], nullptr, 0);
            
            // Parse source address
            uint64_t src = std::stoull(args[1], nullptr, 0);
            
            // Parse size
            size_t size = std::stoull(args[2], nullptr, 0);
            
            if (size == 0) {
                printError("Size must be greater than 0.");
                return;
            }
            
            if (size > 1024 * 1024) { // 1MB limit
                printError("Size must be less than 1MB.");
                return;
            }
            
            // Copy memory using VM's memory subsystem
            MemorySubsystem& memSys = m_vm->getMemorySubsystem();
            std::vector<uint8_t> tempBuffer(size);
            
            // Read from source
            for (size_t i = 0; i < size; ++i) {
                tempBuffer[i] = memSys.read<uint8_t>(MemorySpace::GLOBAL, src + i);
            }
            
            // Write to destination
            for (size_t i = 0; i < size; ++i) {
                memSys.write<uint8_t>(MemorySpace::GLOBAL, dest + i, tempBuffer[i]);
            }
            
            std::ostringstream oss;
            oss << "Copied " << size << " bytes from 0x" << std::hex << src 
                << " to 0x" << dest << std::dec;
            printMessage(oss.str());
        } catch (...) {
            printError("Invalid address or size format.");
        }
    }

    // Write command - write a value to memory
    void writeCommand(const std::vector<std::string>& args) {
        if (args.size() < 2) {
            printError("Usage: write <address> <value>");
            printError("Example: write 0x10000 42");
            return;
        }
        
        try {
            // Parse address
            uint64_t address = std::stoull(args[0], nullptr, 0);
            
            // Parse value
            uint64_t value = std::stoull(args[1], nullptr, 0);
            
            // Validate value is in byte range
            if (value > 255) {
                printError("Value must be between 0 and 255 (one byte).");
                return;
            }
            
            // Write the value to memory
            MemorySubsystem& memSys = m_vm->getMemorySubsystem();
            uint8_t byteValue = static_cast<uint8_t>(value);
            memSys.write<uint8_t>(MemorySpace::GLOBAL, address, byteValue);
            
            std::ostringstream oss;
            oss << "Wrote value " << value << " to address 0x" << std::hex << address << std::dec;
            printMessage(oss.str());
        } catch (...) {
            printError("Invalid address or value format.");
        }
    }

    // Fill command - fill memory with values
    void fillCommand(const std::vector<std::string>& args) {
        if (args.size() < 3) {
            printError("Usage: fill <address> <count> <value1> [value2] ...");
            printError("Example: fill 0x10000 4 1 2 3 4");
            return;
        }
        
        try {
            // Parse address
            uint64_t address = std::stoull(args[0], nullptr, 0);
            
            // Parse count
            size_t count = std::stoull(args[1], nullptr, 0);
            
            // Check if we have enough values
            if (args.size() < count + 2) {
                printError("Not enough values provided for the specified count.");
                return;
            }
            
            // Validate count
            if (count == 0) {
                printError("Count must be greater than 0.");
                return;
            }
            
            if (count > 1024) { // Limit to 1KB
                printError("Count must be less than 1024.");
                return;
            }
            
            // Parse values
            std::vector<uint8_t> values(count);
            for (size_t i = 0; i < count; ++i) {
                uint64_t value = std::stoull(args[i + 2], nullptr, 0);
                if (value > 255) {
                    std::ostringstream oss;
                    oss << "Value " << value << " at position " << i << " is out of range (0-255).";
                    printError(oss.str());
                    return;
                }
                values[i] = static_cast<uint8_t>(value);
            }
            
            // Write values to memory
            MemorySubsystem& memSys = m_vm->getMemorySubsystem();
            for (size_t i = 0; i < count; ++i) {
                memSys.write<uint8_t>(MemorySpace::GLOBAL, address + i, values[i]);
            }
            
            std::ostringstream oss;
            oss << "Filled " << count << " bytes at address 0x" << std::hex << address << std::dec;
            printMessage(oss.str());
        } catch (...) {
            printError("Invalid address, count, or value format.");
        }
    }

    // Memory command - view memory contents
    void memoryCommand(const std::vector<std::string>& args) {
        if (args.size() < 2) {
            printError("Usage: memory <address> <size>");
            printError("Example: memory 0x10000 20");
            return;
        }
        
        try {
            // Parse address
            uint64_t address = std::stoull(args[0], nullptr, 0);
            
            // Parse size
            size_t size = std::stoull(args[1], nullptr, 0);
            
            if (size == 0) {
                printError("Size must be greater than 0.");
                return;
            }
            
            if (size > 1024 * 1024) { // 1MB limit
                printError("Size must be less than 1MB.");
                return;
            }
            
            // Read memory using VM's memory subsystem
            MemorySubsystem& memSys = m_vm->getMemorySubsystem();
            
            std::ostringstream oss;
            oss << "\nMemory at 0x" << std::hex << std::setfill('0') << address 
                << " (" << std::dec << size << " bytes):\n";
            
            // Display memory in hex dump format (16 bytes per line)
            for (size_t offset = 0; offset < size; offset += 16) {
                // Address
                oss << "0x" << std::hex << std::setw(8) << std::setfill('0') 
                    << (address + offset) << ": ";
                
                // Hex bytes
                size_t lineSize = std::min<size_t>(16, size - offset);
                std::vector<uint8_t> lineData(lineSize);
                
                for (size_t i = 0; i < lineSize; ++i) {
                    uint8_t byte = memSys.read<uint8_t>(MemorySpace::GLOBAL, address + offset + i);
                    lineData[i] = byte;
                    oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte << " ";
                }
                
                // Padding if last line
                for (size_t i = lineSize; i < 16; ++i) {
                    oss << "   ";
                }
                
                // ASCII representation
                oss << " |";
                for (size_t i = 0; i < lineSize; ++i) {
                    char c = lineData[i];
                    oss << (isprint(c) ? c : '.');
                }
                oss << "|";
                
                oss << "\n";
            }
            
            // Decode as int32 values if size is multiple of 4
            if (size >= 4 && size % 4 == 0) {
                oss << "\nDecoded as int32_t values:\n";
                for (size_t offset = 0; offset < size; offset += 4) {
                    int32_t value = memSys.read<int32_t>(MemorySpace::GLOBAL, address + offset);
                    oss << "  [0x" << std::hex << std::setw(8) << std::setfill('0') 
                        << (address + offset) << "] " << std::dec << value 
                        << " (0x" << std::hex << std::setw(8) << std::setfill('0') << value << ")\n";
                }
            }
            
            printMessage(oss.str());
        } catch (...) {
            printError("Invalid address or size format, or memory read error.");
        }
    }

    // Helper function to parse parameter value based on PTX type
    bool parseParameterValue(const std::string& valueStr, const PTXParameter& paramDef, 
                            std::vector<uint8_t>& paramData) {
        try {
            paramData.resize(paramDef.size);
            
            // Check if it's a pointer type (.u64 or .s64)
            if (paramDef.isPointer) {
                // Parse as device memory address
                uint64_t addr = std::stoull(valueStr, nullptr, 0);
                std::memcpy(paramData.data(), &addr, sizeof(uint64_t));
                return true;
            }
            
            // Handle scalar types based on PTX type
            if (paramDef.type == ".u32") {
                uint32_t value = std::stoul(valueStr, nullptr, 0);
                std::memcpy(paramData.data(), &value, sizeof(uint32_t));
                return true;
            } else if (paramDef.type == ".s32") {
                int32_t value = std::stoi(valueStr, nullptr, 0);
                std::memcpy(paramData.data(), &value, sizeof(int32_t));
                return true;
            } else if (paramDef.type == ".f32") {
                float value = std::stof(valueStr);
                std::memcpy(paramData.data(), &value, sizeof(float));
                return true;
            } else if (paramDef.type == ".f64") {
                double value = std::stod(valueStr);
                std::memcpy(paramData.data(), &value, sizeof(double));
                return true;
            } else if (paramDef.type == ".u64") {
                uint64_t value = std::stoull(valueStr, nullptr, 0);
                std::memcpy(paramData.data(), &value, sizeof(uint64_t));
                return true;
            } else if (paramDef.type == ".s64") {
                int64_t value = std::stoll(valueStr, nullptr, 0);
                std::memcpy(paramData.data(), &value, sizeof(int64_t));
                return true;
            } else if (paramDef.type == ".u16") {
                uint16_t value = static_cast<uint16_t>(std::stoul(valueStr, nullptr, 0));
                std::memcpy(paramData.data(), &value, sizeof(uint16_t));
                return true;
            } else if (paramDef.type == ".s16") {
                int16_t value = static_cast<int16_t>(std::stoi(valueStr, nullptr, 0));
                std::memcpy(paramData.data(), &value, sizeof(int16_t));
                return true;
            } else if (paramDef.type == ".u8") {
                uint8_t value = static_cast<uint8_t>(std::stoul(valueStr, nullptr, 0));
                paramData[0] = value;
                return true;
            } else if (paramDef.type == ".s8") {
                int8_t value = static_cast<int8_t>(std::stoi(valueStr, nullptr, 0));
                paramData[0] = value;
                return true;
            }
            
            // Unknown type
            return false;
        } catch (...) {
            return false;
        }
    }

    // Launch command - launch a kernel with parameters
    void launchCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            printError("Usage: launch <kernel_name> [param1] [param2] ...");
            printMessage("");
            printMessage("Parameters are automatically typed based on the PTX kernel signature:");
            printMessage("  - No parameters: Just kernel name (e.g., launch myKernel)");
            printMessage("  - Pointer types (.u64): Pass device address (e.g., 0x10000)");
            printMessage("  - Scalar types (.u32, .f32, etc.): Pass value directly (e.g., 1024, 2.5)");
            printMessage("");
            printMessage("Example 1 (no parameters):");
            printMessage("  launch testKernel");
            printMessage("");
            printMessage("Example 2 (pointers only):");
            printMessage("  launch vecAdd 0x10000 0x10020 0x10040");
            printMessage("");
            printMessage("Example 3 (mixed pointers and scalars):");
            printMessage("  launch scaleArray 0x10000 1024 2.5");
            printMessage("  (ptr to data, size N=1024, scale factor=2.5)");
            printMessage("");
            printMessage("Type 'help launch' for complete workflow examples.");
            return;
        }
        
        // First argument is the kernel name
        std::string kernelName = args[0];
        
        // Get the parsed PTX program to determine parameter types
        if (!m_vm->hasProgram()) {
            printError("No PTX program loaded. Use 'load' first.");
            return;
        }
        
        const PTXExecutor& executor = m_vm->getExecutor();
        if (!executor.hasProgramStructure()) {
            printError("PTX program structure not available.");
            return;
        }
        
        const PTXProgram& program = executor.getProgram();
        
        // Find the kernel function
        const PTXFunction* kernel = nullptr;
        for (const auto& func : program.functions) {
            if (func.isEntry && func.name == kernelName) {
                kernel = &func;
                break;
            }
        }
        
        if (kernel == nullptr) {
            printError("Kernel '" + kernelName + "' not found in loaded PTX program.");
            printMessage("Available kernels:");
            for (const auto& func : program.functions) {
                if (func.isEntry) {
                    printMessage("  - " + func.name);
                }
            }
            return;
        }
        
        // Check parameter count
        size_t expectedParams = kernel->parameters.size();
        size_t providedParams = args.size() - 1; // Exclude kernel name
        
        if (providedParams != expectedParams) {
            std::ostringstream oss;
            oss << "Parameter count mismatch: expected " << expectedParams 
                << ", got " << providedParams;
            printError(oss.str());
            printMessage("");
            if (expectedParams == 0) {
                printMessage("Kernel signature: " + kernelName + "()  // No parameters");
                printMessage("Usage: launch " + kernelName);
            } else {
                printMessage("Kernel signature: " + kernelName + "(");
                for (size_t i = 0; i < kernel->parameters.size(); ++i) {
                    const auto& param = kernel->parameters[i];
                    std::ostringstream poss;
                    poss << "  [" << i << "] " << param.type << " " << param.name;
                    if (param.isPointer) {
                        poss << " (pointer - needs device address)";
                    } else {
                        poss << " (scalar - needs value)";
                    }
                    printMessage(poss.str());
                }
                printMessage(")");
            }
            return;
        }
        
        // Parse and prepare parameters based on their PTX types
        printMessage("");
        if (expectedParams == 0) {
            printMessage("Launching kernel with no parameters");
        } else {
            printMessage("Parsing kernel parameters:");
        }
        std::vector<std::vector<uint8_t>> parameterData;
        std::vector<void*> kernelParams;
        
        for (size_t i = 0; i < kernel->parameters.size(); ++i) {
            const PTXParameter& paramDef = kernel->parameters[i];
            const std::string& valueStr = args[i + 1];
            
            std::vector<uint8_t> paramData;
            if (!parseParameterValue(valueStr, paramDef, paramData)) {
                std::ostringstream oss;
                oss << "Failed to parse parameter " << i << " ('" << paramDef.name 
                    << "' of type " << paramDef.type << ") from value: " << valueStr;
                printError(oss.str());
                return;
            }
            
            // Display parameter info
            std::ostringstream poss;
            poss << "  [" << i << "] " << paramDef.name << " (" << paramDef.type << "): ";
            
            if (paramDef.isPointer) {
                uint64_t addr;
                std::memcpy(&addr, paramData.data(), sizeof(uint64_t));
                poss << "device address 0x" << std::hex << addr << std::dec;
            } else if (paramDef.type == ".u32") {
                uint32_t value;
                std::memcpy(&value, paramData.data(), sizeof(uint32_t));
                poss << "value " << value;
            } else if (paramDef.type == ".s32") {
                int32_t value;
                std::memcpy(&value, paramData.data(), sizeof(int32_t));
                poss << "value " << value;
            } else if (paramDef.type == ".f32") {
                float value;
                std::memcpy(&value, paramData.data(), sizeof(float));
                poss << "value " << value;
            } else if (paramDef.type == ".f64") {
                double value;
                std::memcpy(&value, paramData.data(), sizeof(double));
                poss << "value " << value;
            } else {
                poss << "value (binary)";
            }
            
            printMessage(poss.str());
            
            // Store parameter data and create pointer to it
            parameterData.push_back(std::move(paramData));
            kernelParams.push_back(parameterData.back().data());
        }
        
        // Launch the kernel with default grid/block dimensions
        printMessage("");
        printMessage("Launching kernel: " + kernelName);
        printMessage("Grid dimensions: 1 x 1 x 1");
        printMessage("Block dimensions: 32 x 1 x 1");
        
        try {
            PTXExecutor& executor = m_vm->getExecutor();
            
            // Copy kernel parameters to parameter memory
            if (!kernelParams.empty() && executor.hasProgramStructure()) {
                const PTXProgram& program = executor.getProgram();
                
                if (!program.functions.empty()) {
                    const PTXFunction& entryFunc = program.functions[0];
                    MemorySubsystem& mem = executor.getMemorySubsystem();
                    
                    printMessage("Setting up " + std::to_string(entryFunc.parameters.size()) + " kernel parameters...");
                    
                    // Copy each parameter to parameter memory
                    size_t offset = 0;
                    for (size_t i = 0; i < entryFunc.parameters.size(); ++i) {
                        const PTXParameter& param = entryFunc.parameters[i];
                        
                        if (kernelParams[i] != nullptr) {
                            std::ostringstream poss;
                            poss << "  Parameter " << i << " (" << param.name << "): "
                                 << "type=" << param.type << ", size=" << param.size 
                                 << ", offset=" << offset;
                            printMessage(poss.str());
                            
                            // Copy parameter data to parameter memory (base address 0x0)
                            const uint8_t* paramData = static_cast<const uint8_t*>(kernelParams[i]);
                            for (size_t j = 0; j < param.size; ++j) {
                                mem.write<uint8_t>(MemorySpace::PARAMETER, 
                                                  offset + j, 
                                                  paramData[j]);
                            }
                        }
                        
                        offset += param.size;
                    }
                    
                    printMessage("Kernel parameters successfully copied to parameter memory");
                }
            }

            // Execute kernel
            bool success = m_vm->run();
            
            if (success) {
                printMessage("");
                printMessage("✓ Kernel launched successfully");
                printMessage("Use 'memory <address> <size>' to view results");
            } else {
                printError("✗ Kernel execution failed");
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "✗ Kernel launch error: " << e.what();
            printError(oss.str());
        } catch (...) {
            printError("✗ Unknown kernel launch error");
        }
    }

    // Quit command - exit the VM
    void quitCommand(const std::vector<std::string>& args) {
        // Ask if user wants to quit
        if (args.empty() || args[0] != "--force") {
            std::cout << "Are you sure you want to quit? (y/n): ";
            char response;
            std::cin >> response;
            
            if (response != 'y' && response != 'Y') {
                return;
            }
        }
        
        // Exit the application
        exit(0);
    }

    // Clear command - clear the screen
    void clearCommand(const std::vector<std::string>& args) {
        // Clear the console screen
        // Note: This works on Windows only
        system("cls");
    }

    // Version command - show version information
    void versionCommand(const std::vector<std::string>& args) {
        printMessage("NVIDIA PTX Virtual Machine", false);
        printMessage("--------------------------");
        printMessage("Version: 1.0");
        printMessage("Build: " __DATE__ " " __TIME__);
        printMessage("Copyright (C) 2023 NVIDIA Corporation. All rights reserved.");
    }

    // Info command - show current VM information
    void infoCommand(const std::vector<std::string>& args) {
        printMessage("Virtual Machine State:", false);
        printMessage("----------------------");
        
        // VM status
        std::ostringstream oss;
        oss << "Status: " << (m_loadedProgram.empty() ? "No program loaded" : "Program loaded");
        printMessage(oss.str());
        
        // Execution state
        oss.str("");
        oss << "Execution state: " << (m_executing ? "Running" : "Stopped");
        printMessage(oss.str());
        
        // Current PC
        oss.str("");
        oss << "Current PC: 0x" << std::hex << m_currentPC << std::dec;
        printMessage(oss.str());
        
        // Performance counters
        oss.str("");
        oss << "Instructions executed: " << m_vm->getPerformanceCounters().getCounterValue(PerformanceCounterIDs::INSTRUCTIONS_EXECUTED);
        printMessage(oss.str());
        
        oss.str("");
        oss << "Total cycles: " << m_vm->getPerformanceCounters().getCounterValue(PerformanceCounterIDs::CYCLES);
        printMessage(oss.str());
        
        oss.str("");
        oss << "Divergent branches: " << m_vm->getPerformanceCounters().getCounterValue(PerformanceCounterIDs::DIVERGENT_BRANCHES);
        printMessage(oss.str());
        
        oss.str("");
        oss << "Register utilization: " << m_vm->getRegisterAllocator().getRegisterUtilization() * 100 << "%";
        printMessage(oss.str());
    }

    // Visualize command - display visualization
    std::string getCommandLine() {
        std::cout << (m_loadedProgram.empty() ? "ptx-vm> " : ("ptx-vm(" + m_loadedProgram + ")> "));
        std::string line;
        std::getline(std::cin, line);
        return line;
    }

    // Print message to console
    void printMessage(const std::string& message, bool addNewline = true) {
        if (addNewline) {
            std::cout << message << std::endl;
        } else {
            std::cout << message;
        }
    }

    // Print error message to console
    void printError(const std::string& message) {
        std::cerr << "Error: " << message << std::endl;
    }

    // Visualize command - display visualization
    void visualizeCommand(const std::vector<std::string>& args) {
        if (m_loadedProgram.empty()) {
            printError("No program loaded. Use 'load' to load a program first.");
            return;
        }
        
        if (args.empty()) {
            printError("Usage: visualize <type>");
            printError("Available types: warp, memory, performance");
            return;
        }
        
        std::string type = args[0];
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        
        if (type == "warp") {
            // Visualize warp execution
            m_vm->visualizeWarps();
        } else if (type == "memory") {
            // Visualize memory usage
            m_vm->visualizeMemory();
        } else if (type == "performance") {
            // Visualize performance counters
            m_vm->visualizePerformance();
        } else {
            std::ostringstream oss;
            oss << "Unknown visualization type: " << type << ". Use 'help visualize' for options.";
            printError(oss.str());
        }
    }


    // Core components
    std::unique_ptr<PTXVM> m_vm;
    
    // Execution state
    std::string m_loadedProgram;
    size_t m_currentPC = 0; // TODO: Update with actual PC tracking
    bool m_executing = false; // TODO: Update with actual execution state
};

CLIInterface::CLIInterface() : pImpl(std::make_unique<Impl>()) {}

CLIInterface::~CLIInterface() = default;

int CLIInterface::run(int argc, char* argv[]) {
    return pImpl->run(argc, argv);
}

# Logging System Documentation

## Overview

PTX-VM includes a comprehensive logging system that allows users to control the verbosity of diagnostic output. The logging system supports four levels of severity: DEBUG, INFO, WARNING, and ERROR.

## Log Levels

The logging system provides four log levels in ascending order of severity:

1. **DEBUG** - Detailed debug information for troubleshooting
   - Use for: Detailed internal state, variable values, execution traces
   - Best for: Development and debugging

2. **INFO** (Default) - General informational messages
   - Use for: Normal program flow, successful operations
   - Best for: Production use, general monitoring

3. **WARNING** - Warning messages for potential issues
   - Use for: Non-critical problems, deprecated features, recoverable errors
   - Best for: Identifying potential problems

4. **ERROR** - Error messages for failures
   - Use for: Critical errors, failed operations, exceptions
   - Best for: Troubleshooting failures

## Default Behavior

By default, the log level is set to **INFO**. This means:
- INFO, WARNING, and ERROR messages will be displayed
- DEBUG messages will be suppressed

## Configuration

### Command Line Options

When starting PTX-VM, you can set the log level using command-line options:

```bash
# Set log level to DEBUG
ptx_vm --log-level debug program.ptx
ptx_vm -l debug program.ptx

# Set log level to INFO (default)
ptx_vm --log-level info program.ptx

# Set log level to WARNING
ptx_vm --log-level warning program.ptx
ptx_vm -l warn program.ptx

# Set log level to ERROR
ptx_vm --log-level error program.ptx
```

### Interactive Mode

Within the interactive CLI, you can change the log level at runtime using the `loglevel` command:

```
ptx-vm> loglevel debug      # Enable all logs
ptx-vm> loglevel info        # Default level
ptx-vm> loglevel warning     # Warnings and errors only
ptx-vm> loglevel error       # Errors only

ptx-vm> loglevel             # Display current log level
```

### Programmatic Configuration

If you're using PTX-VM as a library, you can configure logging in your code:

```cpp
#include "logger.hpp"

// Set log level
Logger::setLogLevel(LogLevel::DEBUG);

// Enable timestamps
Logger::setShowTimestamp(true);

// Enable/disable colored output
Logger::setColorOutput(true);

// Use the logger
Logger::debug("Detailed debug information");
Logger::info("Normal operation message");
Logger::warning("Warning message");
Logger::error("Error message");
```

## Output Format

Log messages are formatted as follows:

```
[LEVEL   ] Message text
```

Examples:
```
[INFO    ] Loading PTX program: example.ptx
[WARNING ] Deprecated feature used
[ERROR   ] Failed to allocate memory
[DEBUG   ] Register value: r0 = 0x12345678
```

### Colored Output

By default, log messages are colored for easier reading:
- DEBUG: Cyan
- INFO: Green
- WARNING: Yellow
- ERROR: Red

Colors can be disabled if needed:
```cpp
Logger::setColorOutput(false);
```

### Timestamps

Timestamps can be enabled to show when each log message was generated:

```cpp
Logger::setShowTimestamp(true);
```

Output with timestamps:
```
[2025-10-30 14:23:45.123] [INFO    ] Program loaded successfully
```

## Best Practices

### For Users

1. **Normal Use**: Keep the default INFO level for regular operation
2. **Debugging**: Use DEBUG level when troubleshooting issues
3. **Production**: Consider WARNING or ERROR for production environments
4. **Performance**: Higher log levels (ERROR) may improve performance by reducing output

### For Developers

1. Use appropriate log levels:
   - `Logger::debug()` - Detailed implementation details, variable dumps
   - `Logger::info()` - High-level operation progress, successful completions
   - `Logger::warning()` - Non-critical issues, fallback actions
   - `Logger::error()` - Failures, exceptions, critical errors

2. Keep messages concise and informative
3. Include relevant context (addresses, values, names)
4. Avoid logging in tight loops (use DEBUG level if necessary)

## Examples

### Example 1: Debug Mode

```bash
ptx_vm -l debug program.ptx
```

Output:
```
[DEBUG   ] Initializing VM subsystems
[DEBUG   ] Register bank initialized with 256 registers
[DEBUG   ] Memory subsystem initialized: global=4GB, shared=48KB
[INFO    ] Loading PTX program: program.ptx
[DEBUG   ] Parsed 42 instructions
[DEBUG   ] Found entry point: _Z6kernelPi
[INFO    ] Program loaded successfully
```

### Example 2: Info Mode (Default)

```bash
ptx_vm program.ptx
```

Output:
```
[INFO    ] Loading PTX program: program.ptx
[INFO    ] Program loaded successfully
[INFO    ] Executing kernel: _Z6kernelPi
[INFO    ] Execution completed in 1234 cycles
```

### Example 3: Error Only Mode

```bash
ptx_vm -l error program.ptx
```

Output (only if errors occur):
```
[ERROR   ] Failed to parse instruction at line 15
[ERROR   ] Execution failed
```

## Integration with Existing Code

The logging system has been integrated throughout PTX-VM:

- **Parser**: Logs parsing progress and errors
- **Executor**: Logs execution state and errors
- **Memory**: Logs allocation/deallocation operations
- **Debugger**: Error messages use logger (debug output remains direct)
- **CLI**: User interaction messages and errors

## Thread Safety

The logging system is thread-safe. Multiple threads can call logging functions simultaneously without data races.

## Help Command

For quick reference in the CLI:

```
ptx-vm> help loglevel

loglevel [level] - Get or set log level

Usage:
  loglevel          - Display current log level
  loglevel debug    - Set log level to DEBUG
  loglevel info     - Set log level to INFO
  loglevel warning  - Set log level to WARNING
  loglevel error    - Set log level to ERROR

Valid levels: debug, info, warning, error
Default level: info
```

## See Also

- [API Documentation](api_documentation.md) - Full API reference
- [CLI Usage Guide](cli_usage_correction.md) - CLI command reference
- [Developer Guide](../docs_dev/developer_guide.md) - Development guidelines

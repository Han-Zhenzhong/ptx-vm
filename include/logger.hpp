#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <ctime>

/**
 * Log levels in ascending order of severity
 */
enum class LogLevel {
    DEBUG = 0,    // Detailed debug information
    INFO = 1,     // General informational messages
    WARNING = 2,  // Warning messages
    ERROR = 3     // Error messages
};

/**
 * Logger class for leveled logging
 * 
 * Provides static methods for logging at different levels.
 * Default log level is INFO - messages with level INFO and above will be printed.
 * 
 * Usage:
 *   Logger::debug("Detailed debug info");
 *   Logger::info("Normal operation message");
 *   Logger::warning("Warning message");
 *   Logger::error("Error message");
 * 
 * Configuration:
 *   Logger::setLogLevel(LogLevel::DEBUG);  // Enable all logs
 *   Logger::setLogLevel(LogLevel::ERROR);  // Only errors
 */
class Logger {
public:
    /**
     * Set the global log level
     * Only messages at or above this level will be printed
     * @param level The minimum log level to display
     */
    static void setLogLevel(LogLevel level);

    /**
     * Get the current log level
     * @return Current log level
     */
    static LogLevel getLogLevel();

    /**
     * Enable or disable timestamps in log output
     * @param enable true to show timestamps, false to hide
     */
    static void setShowTimestamp(bool enable);

    /**
     * Enable or disable colored output (ANSI colors)
     * @param enable true to enable colors, false to disable
     */
    static void setColorOutput(bool enable);

    /**
     * Log a debug message (LogLevel::DEBUG)
     * @param message The message to log
     */
    static void debug(const std::string& message);

    /**
     * Log an info message (LogLevel::INFO)
     * @param message The message to log
     */
    static void info(const std::string& message);

    /**
     * Log a warning message (LogLevel::WARNING)
     * @param message The message to log
     */
    static void warning(const std::string& message);

    /**
     * Log an error message (LogLevel::ERROR)
     * @param message The message to log
     */
    static void error(const std::string& message);

    /**
     * Generic log method with explicit level
     * @param level The log level
     * @param message The message to log
     */
    static void log(LogLevel level, const std::string& message);

    /**
     * Convert log level to string
     * @param level The log level
     * @return String representation of the level
     */
    static std::string levelToString(LogLevel level);

private:
    static LogLevel currentLogLevel;
    static bool showTimestamp;
    static bool colorOutput;
    static std::mutex logMutex;

    /**
     * Internal logging implementation
     * @param level The log level
     * @param message The message to log
     */
    static void logImpl(LogLevel level, const std::string& message);

    /**
     * Get ANSI color code for log level
     * @param level The log level
     * @return ANSI color code string
     */
    static std::string getColorCode(LogLevel level);

    /**
     * Get current timestamp string
     * @return Formatted timestamp
     */
    static std::string getTimestamp();
};

#endif // LOGGER_HPP

#include "logger.hpp"
#include <iomanip>
#include <chrono>

// Initialize static members
LogLevel Logger::currentLogLevel = LogLevel::INFO;  // Default level is INFO
bool Logger::showTimestamp = false;
bool Logger::colorOutput = true;
std::mutex Logger::logMutex;

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(logMutex);
    currentLogLevel = level;
}

LogLevel Logger::getLogLevel() {
    std::lock_guard<std::mutex> lock(logMutex);
    return currentLogLevel;
}

void Logger::setShowTimestamp(bool enable) {
    std::lock_guard<std::mutex> lock(logMutex);
    showTimestamp = enable;
}

void Logger::setColorOutput(bool enable) {
    std::lock_guard<std::mutex> lock(logMutex);
    colorOutput = enable;
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::log(LogLevel level, const std::string& message) {
    logImpl(level, message);
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

void Logger::logImpl(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    // Filter by log level
    if (level < currentLogLevel) {
        return;
    }
    
    std::ostream& out = (level >= LogLevel::ERROR) ? std::cerr : std::cout;
    
    // Build the log message
    std::ostringstream oss;
    
    // Add color code if enabled
    if (colorOutput) {
        oss << getColorCode(level);
    }
    
    // Add timestamp if enabled
    if (showTimestamp) {
        oss << "[" << getTimestamp() << "] ";
    }
    
    // Add level
    oss << "[" << std::setw(7) << levelToString(level) << "] ";
    
    // Add message
    oss << message;
    
    // Reset color if enabled
    if (colorOutput) {
        oss << "\033[0m";  // Reset color
    }
    
    // Output to stream
    out << oss.str() << std::endl;
}

std::string Logger::getColorCode(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "\033[36m";  // Cyan
        case LogLevel::INFO:    return "\033[32m";  // Green
        case LogLevel::WARNING: return "\033[33m";  // Yellow
        case LogLevel::ERROR:   return "\033[31m";  // Red
        default:                return "";
    }
}

std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    struct tm timeinfo;
    
#ifdef _WIN32
    localtime_s(&timeinfo, &time_t_now);
#else
    localtime_r(&time_t_now, &timeinfo);
#endif
    
    oss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}
